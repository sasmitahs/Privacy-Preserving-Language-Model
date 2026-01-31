#!/usr/bin/env python3
"""
MMLU-Pro – DPO-only (DP-SGD & Fixed-PDP-SGD) with category-wise test accuracy.
Dataset is shuffled **once** before any split.
"""

# ============================================================================
# SSL BYPASS (keeps the script runnable on offline nodes)
# ============================================================================
import os, ssl, warnings, random, math, json, logging
from datetime import datetime
ssl._create_default_https_context = ssl._create_unverified_context
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''
os.environ['SSL_CERT_FILE'] = ''
warnings.filterwarnings('ignore', message='Unverified HTTPS request')
try:
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
except Exception:
    pass

# ============================================================================
# Imports
# ============================================================================
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Sampler
from torch.optim import AdamW
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.utils.extmath import randomized_svd
from tqdm import tqdm
from collections import Counter, defaultdict

# ============================================================================
# Logging & helpers
# ============================================================================
def setup_logging(log_file="mmlu_pro_dpo_only.log"):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file, mode='w'), logging.StreamHandler()]
    )
    warnings.filterwarnings("ignore")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)

class FixedBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, drop_last=True, seed=42):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.generator = torch.Generator().manual_seed(seed)

    def __iter__(self):
        if not self.dataset: return iter([])
        n = len(self.dataset)
        idxs = torch.randperm(n, generator=self.generator).tolist()
        batches = [idxs[i:i+self.batch_size] for i in range(0, n, self.batch_size)]
        return iter([b for b in batches if len(b) == self.batch_size or not self.drop_last])

    def __len__(self):
        n = len(self.dataset)
        return n // self.batch_size if self.drop_last else math.ceil(n / self.batch_size)

def trainable_params(model: nn.Module):
    return [p for p in model.parameters() if p.requires_grad]

def flatten_grads(params):
    grads = []
    for p in params:
        if p.grad is not None:
            grads.append(p.grad.view(-1))
        else:
            grads.append(torch.zeros(p.numel(), device=p.device))
    return torch.cat(grads) if grads else torch.tensor([], device="cpu")

# ============================================================================
# MMLU-Pro utilities
# ============================================================================
def valid_example(ex):
    return (
        "options" in ex and "answer_index" in ex and
        isinstance(ex["options"], list) and isinstance(ex["answer_index"], int) and
        0 <= ex["answer_index"] < len(ex["options"]) and
        ex["question"] and all(isinstance(o, str) and o for o in ex["options"])
    )

def format_mmlu_prompt(question: str, options: list) -> str:
    letters = [chr(ord('A')+i) for i in range(len(options))]
    opt_text = "\n".join(f"{l}. {o}" for l, o in zip(letters, options))
    return f"This is a MCQ Question. Think wisely and provide only the single capital letter, e.g., A, B, C corresponding to the correct answer. Question: {question}\nOptions:\n{opt_text}\nAnswer: "

# ============================================================================
# DPO loss helpers
# ============================================================================
class DPOLoss(nn.Module):
    def __init__(self, beta=0.1): super().__init__(); self.beta = beta
    def forward(self, pc, pr, rc, rr):
        logits = self.beta * ((pc - pr) - (rc - rr))
        return -F.logsigmoid(logits).mean()

def get_batch_logps(logits, labels):
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss_fct = nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
    per_token = -loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    per_token = per_token.view(labels.size(0), -1)
    mask = (shift_labels != -100).float()
    return (per_token * mask).sum(-1) / mask.sum(-1).clamp(min=1)

# ============================================================================
# Category-wise evaluation
# ============================================================================
@torch.no_grad()
def evaluate_category_wise(model, loader, tokenizer, device):
    model.eval()
    letters = [chr(ord('A')+i) for i in range(10)]
    completions = [f" {l}" for l in letters]
    token_ids = [tokenizer.encode(c, add_special_tokens=False)[0] if tokenizer.encode(c, add_special_tokens=False) else tokenizer.pad_token_id
                 for c in completions]
    token_ids = torch.tensor(token_ids, device=device)

    stats = defaultdict(lambda: {"correct": 0, "total": 0})

    def get_prob(input_ids, attention_mask, token_id):
        if token_id == tokenizer.pad_token_id: return 0.0
        out = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = out.logits[:, -1, :]
        return F.softmax(logits, dim=-1)[0, token_id].item()

    for batch in loader:
        c_ids = batch[0].to(device)
        c_mask = batch[1].to(device)
        c_lab = batch[2].to(device)
        correct_idx = batch[6].to(device)          # correct answer index
        categories = batch[7].to(device)           # **numeric** category ids

        for i in range(c_ids.size(0)):
            try:
                non_ignore = (c_lab[i] != -100).nonzero(as_tuple=True)[0]
                if len(non_ignore) == 0: continue
                prompt_len = non_ignore[0].item()
                input_ids = c_ids[i][:prompt_len].unsqueeze(0)
                attention_mask = c_mask[i][:prompt_len].unsqueeze(0)

                probs = [get_prob(input_ids, attention_mask, tid) for tid in token_ids]
                pred_idx = np.argmax(probs)
                true_idx = correct_idx[i].item()
                cat = int(categories[i].item())

                stats[cat]["total"] += 1
                if pred_idx == true_idx:
                    stats[cat]["correct"] += 1
            except Exception as e:
                logging.warning(f"Eval error: {e}")

    acc_dict = {cat: s["correct"]/max(s["total"],1) for cat, s in stats.items()}
    overall = sum(s["correct"] for s in stats.values()) / max(sum(s["total"] for s in stats.values()),1)
    return acc_dict, overall

# -------------------------------------------------------------------------
# Add **numeric** category ids to a DPO TensorDataset (8-th tensor)
# -------------------------------------------------------------------------
def add_category_to_dataset(dataset, category_list):
    """
    `category_list` must be a **list of integers**.
    """
    tensors = [t for t in dataset.tensors]
    cat_tensor = torch.tensor(category_list, dtype=torch.long)
    tensors.append(cat_tensor)
    return TensorDataset(*tensors)

# ============================================================================
# DPO dataset builder (keeps raw example for category)
# ============================================================================
def build_dpo_examples_from_hf(ds) -> list:
    examples, skipped = [], Counter()
    for i, item in enumerate(ds):
        if not valid_example(item):
            skipped['invalid'] += 1; continue
        q, opts, ans_idx = item.get("question",""), item.get("options",[]), item.get("answer_index",0)
        if not q or len(opts) != 10:
            skipped['invalid_len'] += 1; continue
        ans_text = opts[ans_idx]
        opts_shuf = opts[:]; random.shuffle(opts_shuf)
        try: correct_idx = opts_shuf.index(ans_text)
        except ValueError:
            skipped['ans_not_in_shuf'] += 1; continue
        letters = [chr(ord('A')+j) for j in range(10)]
        rejected = random.choice([l for l in letters if l != letters[correct_idx]])
        examples.append({
            "prompt": format_mmlu_prompt(q, opts_shuf),
            "chosen": letters[correct_idx],
            "rejected": rejected,
            "correct_idx": correct_idx,
            "raw": item               # keep raw for category
        })
    logging.info(f"DPO examples: {len(examples)} (skipped {dict(skipped)})")
    return examples

def build_dpo_dataset(examples, tokenizer, max_length=512):
    enc, eos = [], tokenizer.eos_token or "<|im_end|>"
    for ex in examples:
        p = ex["prompt"]
        c = ex["chosen"] + eos; r = ex["rejected"] + eos
        enc_c = tokenizer(p + c, truncation=True, padding="max_length",
                          max_length=max_length, return_tensors="pt")
        enc_r = tokenizer(p + r, truncation=True, padding="max_length",
                          max_length=max_length, return_tensors="pt")
        # chosen
        c_ids = enc_c["input_ids"].squeeze(0); c_mask = enc_c["attention_mask"].squeeze(0)
        c_lab = c_ids.clone()
        prompt_len = len(tokenizer(p, truncation=True, max_length=max_length)["input_ids"])
        c_lab[:prompt_len] = -100; c_lab[c_ids == tokenizer.pad_token_id] = -100
        # rejected
        r_ids = enc_r["input_ids"].squeeze(0); r_mask = enc_r["attention_mask"].squeeze(0)
        r_lab = r_ids.clone()
        r_lab[:prompt_len] = -100; r_lab[r_ids == tokenizer.pad_token_id] = -100
        enc.append({
            "c_ids":c_ids, "c_mask":c_mask, "c_lab":c_lab,
            "r_ids":r_ids, "r_mask":r_mask, "r_lab":r_lab,
            "correct_idx": torch.tensor(ex["correct_idx"], dtype=torch.long)
        })
    if not enc: return None
    return TensorDataset(
        torch.stack([e["c_ids"] for e in enc]),
        torch.stack([e["c_mask"] for e in enc]),
        torch.stack([e["c_lab"] for e in enc]),
        torch.stack([e["r_ids"] for e in enc]),
        torch.stack([e["r_mask"] for e in enc]),
        torch.stack([e["r_lab"] for e in enc]),
        torch.stack([e["correct_idx"] for e in enc])
    )

# ============================================================================
# DP-SGD DPO
# ============================================================================
def train_dp_sgd_dpo(model, ref_model, loader, test_loader, tokenizer, device,
                     max_grad_norm, lr, epochs=3, sigma=10.0):
    optimizer = AdamW(trainable_params(model), lr=lr)
    dpo_loss_fn = DPOLoss(beta=0.1)
    tparams = trainable_params(model)
    param_sizes = [p.numel() for p in tparams]
    param_shapes = [p.shape for p in tparams]

    for epoch in range(epochs):
        model.train(); ref_model.eval()
        for batch in tqdm(loader, desc=f"DP-DPO Epoch {epoch+1}"):
            c_ids, c_mask, c_lab = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            r_ids, r_mask, r_lab = batch[3].to(device), batch[4].to(device), batch[5].to(device)

            per_sample_grads = []
            for i in range(c_ids.size(0)):
                model.zero_grad(set_to_none=True)
                c_out = model(input_ids=c_ids[i:i+1], attention_mask=c_mask[i:i+1])
                r_out = model(input_ids=r_ids[i:i+1], attention_mask=r_mask[i:i+1])
                pc = get_batch_logps(c_out.logits, c_lab[i:i+1])
                pr = get_batch_logps(r_out.logits, r_lab[i:i+1])
                with torch.no_grad():
                    rc = get_batch_logps(ref_model(input_ids=c_ids[i:i+1], attention_mask=c_mask[i:i+1]).logits,
                                         c_lab[i:i+1])
                    rr = get_batch_logps(ref_model(input_ids=r_ids[i:i+1], attention_mask=r_mask[i:i+1]).logits,
                                         r_lab[i:i+1])
                loss = dpo_loss_fn(pc, pr, rc, rr)
                if loss > 100: loss = loss.clamp(max=100)
                loss.backward()
                g = flatten_grads(tparams)
                g_norm = torch.norm(g).item()
                if g_norm > 0: g = g * min(1.0, max_grad_norm / g_norm)
                per_sample_grads.append(g)
                model.zero_grad(set_to_none=True)

            if not per_sample_grads: continue
            g_t = torch.stack(per_sample_grads).mean(0)
            noise = torch.normal(0, sigma * max_grad_norm, size=g_t.shape, device=device)
            g_noisy = g_t + noise
            idx = 0
            for p, sz, sh in zip(tparams, param_sizes, param_shapes):
                p.grad = g_noisy[idx:idx+sz].view(sh).type_as(p)
                idx += sz
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        _, overall = evaluate_category_wise(model, test_loader, tokenizer, device)
        logging.info(f"DP-DPO E{epoch+1}: Overall Acc={overall:.4f}")
    return overall

# ============================================================================
# Fixed-PDP-SGD DPO
# ============================================================================
def train_pdp_sgd_dpo_fixed(model, ref_model, private_loader, test_loader, tokenizer, device,
                            max_grad_norm, lr, V_k, total_steps=1000, sigma=10.0):
    optimizer = AdamW(trainable_params(model), lr=lr)
    dpo_loss_fn = DPOLoss(beta=0.1)
    tparams = trainable_params(model)
    param_sizes = [p.numel() for p in tparams]
    private_iter = iter(private_loader)
    step = 0
    pbar = tqdm(total=total_steps, desc="PDP-DPO (fixed)")
    while step < total_steps:
        step += 1; pbar.update(1)
        try: batch = next(private_iter)
        except StopIteration:
            private_iter = iter(private_loader)
            batch = next(private_iter)

        c_ids, c_mask, c_lab = batch[0].to(device), batch[1].to(device), batch[2].to(device)
        r_ids, r_mask, r_lab = batch[3].to(device), batch[4].to(device), batch[5].to(device)

        per_sample_grads = []
        for i in range(c_ids.size(0)):
            model.zero_grad(set_to_none=True)
            c_out = model(input_ids=c_ids[i:i+1], attention_mask=c_mask[i:i+1])
            r_out = model(input_ids=r_ids[i:i+1], attention_mask=r_mask[i:i+1])
            pc = get_batch_logps(c_out.logits, c_lab[i:i+1])
            pr = get_batch_logps(r_out.logits, r_lab[i:i+1])
            with torch.no_grad():
                rc = get_batch_logps(ref_model(input_ids=c_ids[i:i+1], attention_mask=c_mask[i:i+1]).logits,
                                     c_lab[i:i+1])
                rr = get_batch_logps(ref_model(input_ids=r_ids[i:i+1], attention_mask=r_mask[i:i+1]).logits,
                                     r_lab[i:i+1])
            loss = dpo_loss_fn(pc, pr, rc, rr)
            if loss > 100: loss = loss.clamp(max=100)
            loss.backward()
            g = flatten_grads(tparams)
            g_norm = torch.norm(g).item()
            if g_norm > 0: g = g * min(1.0, max_grad_norm / g_norm)
            per_sample_grads.append(g)
            model.zero_grad(set_to_none=True)

        if not per_sample_grads: continue
        g_t = torch.stack(per_sample_grads).mean(0)
        noise = torch.normal(0, sigma * max_grad_norm, size=g_t.shape, device=device)
        g_noisy = g_t + noise
        g_tilde = V_k @ (V_k.T @ g_noisy)
        idx = 0
        for p, sz in zip(tparams, param_sizes):
            p.grad = g_tilde[idx:idx+sz].view(p.shape)
            idx += sz
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if step % 100 == 0:
            _, overall = evaluate_category_wise(model, test_loader, tokenizer, device)
            logging.info(f"PDP-DPO Step {step}: Overall Acc={overall:.4f}")
    pbar.close()
    return evaluate_category_wise(model, test_loader, tokenizer, device)[1]

# ============================================================================
# Gradient matrix for DPO (used to compute V_k on public data)
# ============================================================================
def compute_gradient_matrix_dpo(model, ref_model, loader, device, max_grad_norm):
    grads, tparams = [], trainable_params(model)
    dpo_loss_fn = DPOLoss(beta=0.1)
    for batch in loader:
        for i in range(batch[0].size(0)):
            model.zero_grad(set_to_none=True)
            c_out = model(input_ids=batch[0][i:i+1].to(device), attention_mask=batch[1][i:i+1].to(device))
            r_out = model(input_ids=batch[3][i:i+1].to(device), attention_mask=batch[4][i:i+1].to(device))
            pc = get_batch_logps(c_out.logits, batch[2][i:i+1].to(device))
            pr = get_batch_logps(r_out.logits, batch[5][i:i+1].to(device))
            with torch.no_grad():
                rc = get_batch_logps(ref_model(input_ids=batch[0][i:i+1].to(device), attention_mask=batch[1][i:i+1].to(device)).logits,
                                     batch[2][i:i+1].to(device))
                rr = get_batch_logps(ref_model(input_ids=batch[3][i:i+1].to(device), attention_mask=batch[4][i:i+1].to(device)).logits,
                                     batch[5][i:i+1].to(device))
            loss = dpo_loss_fn(pc, pr, rc, rr)
            if loss is None: continue
            loss.backward()
            g = flatten_grads(tparams)
            g_norm = torch.norm(g).item()
            if g_norm > 0: g = g * min(1.0, max_grad_norm / g_norm)
            grads.append(g.cpu())
            model.zero_grad(set_to_none=True)
    return torch.stack(grads).numpy()

def gradient_subspace_distance(G1, G2, k=20):
    _, _, V1 = randomized_svd(G1, n_components=k, random_state=42)
    _, _, V2 = randomized_svd(G2, n_components=k, random_state=42)
    M = V1 @ V2.T
    s = np.linalg.svd(M, compute_uv=False)
    return np.sqrt(k - np.sum(s ** 2))

# ============================================================================
# LoRA model
# ============================================================================
def build_qwen2_lora(local_path, fp16=False):
    model = AutoModelForCausalLM.from_pretrained(
        local_path,
        local_files_only=True,
        dtype=torch.float16 if fp16 else torch.bfloat16,
        trust_remote_code=True
    )
    if model.config.pad_token_id is None:
        model.config.pad_token_id = model.config.eos_token_id
    cfg = LoraConfig(
        r=8, lora_alpha=8,
        target_modules=["q_proj","k_proj","v_proj","o_proj"],
        lora_dropout=0.1, bias="none", task_type=TaskType.CAUSAL_LM
    )
    return get_peft_model(model, cfg)

def now_ts(): return datetime.now().strftime("%Y%m%dT%H%M%S")

# ============================================================================
# MAIN
# ============================================================================
# ... [ALL PREVIOUS CODE UP TO main() REMAINS UNCHANGED] ...

# ============================================================================
# MAIN
# ============================================================================
def main():
    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", default="./mmlu_pro_dataset", help="Local MMLU-Pro folder")
    parser.add_argument("--model_dir",    default="./qwen_model",        help="Local Qwen folder")
    parser.add_argument("--batch_size",   type=int,   default=32)
    parser.add_argument("--lr",           type=float, default=1e-3)
    parser.add_argument("--max_grad_norm",type=float, default=5.0)
    parser.add_argument("--sigma",        type=float, default=10.0)
    parser.add_argument("--epochs",       type=int,   default=3)
    parser.add_argument("--k",            type=int,   default=20)
    parser.add_argument("--max_length",   type=int,   default=512)
    parser.add_argument("--seed",         type=int,   default=42)
    args = parser.parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # ------------------- Load & shuffle once -------------------
    ds = load_from_disk(args.dataset_dir)               # official test split
    N = len(ds)
    logging.info(f"Loaded MMLU-Pro test split: {N} examples")

    rng = np.random.default_rng(args.seed)
    shuffled_idx = rng.permutation(N).tolist()
    ds = ds.select(shuffled_idx)

    # ------------------- Deterministic split -------------------
    test_N   = N // 16
    private_N = 10 * test_N
    public_N  = 5 * test_N

    private = ds.select(range(private_N))
    public  = ds.select(range(private_N, private_N + public_N))
    test    = ds.select(range(private_N + public_N, private_N + public_N + test_N))

    # ------------------- Tokenizer -------------------
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    # ------------------- Build DPO datasets -------------------
    private_dpo_ex = build_dpo_examples_from_hf(private)
    test_dpo_ex    = build_dpo_examples_from_hf(test)

    private_dpo_ds = build_dpo_dataset(private_dpo_ex, tokenizer, args.max_length)
    test_dpo_ds    = build_dpo_dataset(test_dpo_ex, tokenizer, args.max_length)

    if any(x is None for x in [private_dpo_ds, test_dpo_ds]):
        raise RuntimeError("Failed to build DPO datasets")

    # ---- map category **strings → integers** and add to test dataset ----
    unique_cats = sorted({ex["raw"]["category"] for ex in test_dpo_ex})
    cat_to_id = {cat: idx for idx, cat in enumerate(unique_cats)}
    test_categories = [cat_to_id[ex["raw"]["category"]] for ex in test_dpo_ex]   # **integers**
    test_dpo_ds = add_category_to_dataset(test_dpo_ds, test_categories)

    num_batches = len(private_dpo_ds) // args.batch_size
    total_steps = args.epochs * num_batches

    private_dpo_loader = DataLoader(private_dpo_ds,
                                    batch_sampler=FixedBatchSampler(private_dpo_ds, args.batch_size))
    test_loader = DataLoader(test_dpo_ds,
                             batch_sampler=FixedBatchSampler(test_dpo_ds, args.batch_size, drop_last=False))

    # ------------------- Base model & frozen reference -------------------
    init_model = build_qwen2_lora(args.model_dir).to(device)
    ref_model  = build_qwen2_lora(args.model_dir).to(device)
    ref_model.load_state_dict(init_model.state_dict())
    ref_model.eval()

    # ==================== DP-SGD DPO ====================
    model_dp = build_qwen2_lora(args.model_dir).to(device)
    model_dp.load_state_dict(init_model.state_dict())
    dp_overall = train_dp_sgd_dpo(model_dp, ref_model, private_dpo_loader, test_loader,
                                  tokenizer, device, args.max_grad_norm, args.lr,
                                  epochs=args.epochs, sigma=args.sigma)

    # ==================== PDP-SGD DPO (5 public subsets) ====================
    results = []
    num_public_subsets = 5
    subset_size = len(public) // num_public_subsets

    for i in range(num_public_subsets):
        start = i * subset_size
        end   = (i + 1) * subset_size if i < num_public_subsets - 1 else len(public)
        subset = public.select(range(start, end))
        name = f"subset_{i}"

        dpo_ex = build_dpo_examples_from_hf(subset)
        dpo_ds = build_dpo_dataset(dpo_ex, tokenizer, args.max_length)
        if dpo_ds is None: continue
        public_dpo_loader = DataLoader(dpo_ds,
                                       batch_sampler=FixedBatchSampler(dpo_ds, args.batch_size))

        # ----- GSD (private vs public) -----
        G_priv = compute_gradient_matrix_dpo(init_model, ref_model, private_dpo_loader, device, args.max_grad_norm)
        G_pub  = compute_gradient_matrix_dpo(init_model, ref_model, public_dpo_loader, device, args.max_grad_norm)
        gsd = gradient_subspace_distance(G_priv, G_pub, args.k)

        # ----- Build V_k from public -----
        _, _, Vt = randomized_svd(G_pub, n_components=args.k, random_state=42)
        V_k = torch.from_numpy(Vt.T).float().to(device)

        # ----- PDP-DPO training -----
        model_pdp = build_qwen2_lora(args.model_dir).to(device)
        model_pdp.load_state_dict(init_model.state_dict())
        pdp_overall = train_pdp_sgd_dpo_fixed(
            model=model_pdp, ref_model=ref_model,
            private_loader=private_dpo_loader, test_loader=test_loader,
            tokenizer=tokenizer, device=device,
            max_grad_norm=args.max_grad_norm, lr=args.lr,
            V_k=V_k, total_steps=total_steps, sigma=args.sigma)

        # ----- Final category-wise numbers -----
        cat_dp,  _ = evaluate_category_wise(model_dp,  test_loader, tokenizer, device)
        cat_pdp, _ = evaluate_category_wise(model_pdp, test_loader, tokenizer, device)

        results.append({
            "subset": name,
            "gsd": gsd,
            "dp_overall": dp_overall,
            "pdp_overall": pdp_overall,
            "cat_dp": cat_dp,
            "cat_pdp": cat_pdp,
            "cat_name": {v: k for k, v in cat_to_id.items()}   # for readability in JSON
        })

    # ==================== Pretty table: DP and PDP separately per category ====================

    print("\n" + "="*200)
    print("DPO-ONLY CATEGORY-WISE ACCURACY (DP-SGD vs PDP-SGD per category, over 5 public subsets)")
    print("="*200)

    # Sort category names alphabetically
    all_cat_names = sorted(cat_to_id.keys())
    cat_id_order = [cat_to_id[name] for name in all_cat_names]

    # Header: DP-<cat> | PDP-<cat> for each category
    header_parts = ["subset"]
    for name in all_cat_names:
        short = name[:10]  # truncate for alignment
        header_parts.append(f"DP-{short}")
        header_parts.append(f"PDP-{short}")
    header_parts += ["DP-Overall", "PDP-Overall"]
    header = "| " + " | ".join([f"{x:^11}" for x in header_parts]) + " |"
    print(header)

    separator = "|--------|" + "------------|------------|" * len(all_cat_names) + "------------|-------------|"
    print(separator)

    for r in results:
        row = f"| {r['subset']:<6} |"
        for cid in cat_id_order:
            dp_acc  = r["cat_dp"].get(cid, 0.0)
            pdp_acc = r["cat_pdp"].get(cid, 0.0)
            row += f" {dp_acc:6.2%} | {pdp_acc:6.2%} |"
        row += f" {r['dp_overall']:7.2%} | {r['pdp_overall']:7.2%} |"
        print(row)

    print("="*200)

    # Save full per-category dicts
    with open(f"dpo_only_category_results_separate_{now_ts()}.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
