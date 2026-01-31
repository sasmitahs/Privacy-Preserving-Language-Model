#!/usr/bin/env python3
"""
Hindi-only Private Training: SFT + DPO with DP vs PDP
- Private data: Hindi only (full test split used as private training)
- Public data: English only (used solely for subspace estimation in PDP)
- Test set: Hindi dev split only
- Reports accuracy on Hindi test after SFT and after DPO
- After every epoch (last batch), reports distribution of correct options and model predictions in training batch
- Log filename includes public subset size and batch size
"""

import os
import gc
import random
import logging
from collections import defaultdict, Counter
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.utils.extmath import randomized_svd
import pyarrow as pa

# ==================== CONFIG ====================
HOME = os.path.expanduser("~")
MODEL_PATH = os.path.join(HOME, "hf_downloads", "llama-3.2-1b-instruct")
DATASET_ROOT = os.path.join(HOME, "hf_downloads", "global_mmlu_dataset", "CohereLabs___global-mmlu")
os.makedirs("logs", exist_ok=True)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
assert device.type == "cuda", "GPU required!"
NUM_GPUS = torch.cuda.device_count()
print(f"Detected {NUM_GPUS} GPU(s)")

DTYPE = torch.bfloat16

# Key hyperparameters (will appear in log filename)
PUBLIC_SUBSET_SIZE = 600          # English examples per PDP fold (for subspace estimation)
BATCH_SIZE = 32                   # Training batch size
NUM_FOLDS = 5
SUBSPACE_K = 20
SEED = 42
MAX_GRAD_NORM = 1.0               # Gradient clipping norm for DP-SGD
SIGMA = 1.0                       # Noise multiplier for DP
LR = 1e-4
EPOCHS = 3

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

logger = logging.getLogger(__name__)

# ==================== DATA LOADING ====================
def load_split(lang, split_name):
    path = os.path.join(DATASET_ROOT, lang, "0.0.0")
    if not os.path.isdir(path): return None
    for d in os.listdir(path):
        full = os.path.join(path, d)
        if not os.path.isdir(full): continue
        fpath = os.path.join(full, f"global-mmlu-{split_name}.arrow")
        if os.path.exists(fpath):
            try:
                with open(fpath, 'rb') as f:
                    reader = pa.ipc.open_stream(f)
                    table = pa.Table.from_batches(list(reader))
                    return table.to_pydict() if table.num_rows > 0 else None
            except:
                pass
    return None

def load_hindi_data():
    hi_test = load_split("hi", "test")
    hi_dev = load_split("hi", "dev")
    en_test = load_split("en", "test")
    if not all([hi_test, hi_dev, en_test]):
        raise ValueError("Missing Hindi or English data")
    logger.info(f"Hindi test (private train): {len(hi_test['question'])} | "
                f"Hindi dev (test): {len(hi_dev['question'])} | "
                f"English test (public): {len(en_test['question'])}")
    return hi_test, hi_dev, en_test

# ==================== TOKENIZATION HELPERS ====================
def build_sft_examples(examples):
    pairs = []
    n = len(examples["question"])
    for i in range(n):
        q = str(examples["question"][i] or "").strip()
        if not q: continue
        opts = []
        for c in "abcd":
            val = examples.get(f"option_{c}", [None]*n)[i]
            if val is None or str(val).strip() == "": break
            opts.append(str(val).strip())
        if len(opts) != 4: continue
        ans = str(examples.get("answer", [None]*n)[i] or "").strip().upper()
        if ans not in "ABCD": continue
        prompt = f"Question: {q}\n" + "\n".join(f"{chr(65+j)}. {opts[j]}" for j in range(4)) + "\nAnswer:"
        pairs.append({"prompt": prompt, "completion": f" {ans}", "correct_letter": ans})
    return pairs

def build_dpo_pairs(examples):
    pairs = []
    n = len(examples["question"])
    for i in range(n):
        q = str(examples["question"][i] or "").strip()
        if not q: continue
        opts = []
        for c in "abcd":
            val = examples.get(f"option_{c}", [None]*n)[i]
            if val is None or str(val).strip() == "": break
            opts.append(str(val).strip())
        if len(opts) != 4: continue
        ans = str(examples.get("answer", [None]*n)[i] or "").strip().upper()
        if ans not in "ABCD": continue
        correct_idx = ord(ans) - 65
        wrong_idx = random.choice([j for j in range(4) if j != correct_idx])
        prompt = f"Question: {q}\n" + "\n".join(f"{chr(65+j)}. {opts[j]}" for j in range(4)) + "\nAnswer:"
        pairs.append({
            "prompt": prompt,
            "chosen": f" {ans}",
            "rejected": f" {chr(65 + wrong_idx)}",
            "correct_idx": correct_idx,
            "correct_letter": ans
        })
    return pairs

def tokenize_sft(pairs, tokenizer, max_len=512):
    if not pairs: return None
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    input_ids, attention_masks, labels = [], [], []
    correct_idxs = []
    for p in pairs:
        full_text = p["prompt"] + p["completion"] + tokenizer.eos_token
        enc = tokenizer(full_text, truncation=True, max_length=max_len)
        ids = torch.tensor(enc["input_ids"])
        mask = torch.tensor(enc["attention_mask"])
        lab = ids.clone()
        prompt_len = len(tokenizer(p["prompt"], add_special_tokens=False)["input_ids"])
        lab[:prompt_len] = -100
        input_ids.append(ids)
        attention_masks.append(mask)
        labels.append(lab)
        correct_idxs.append(ord(p["correct_letter"]) - 65)
    max_l = max(len(x) for x in input_ids)
    def pad(t): return torch.cat([t, torch.full((max_l - len(t),), pad_id, dtype=t.dtype)]) if len(t) < max_l else t
    return TensorDataset(
        torch.stack([pad(x) for x in input_ids]),
        torch.stack([pad(x) for x in attention_masks]),
        torch.stack([pad(x) for x in labels]),
        torch.tensor(correct_idxs)
    )

def tokenize_dpo(pairs, tokenizer, max_len=512):
    if not pairs: return None
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    data = defaultdict(list)
    for p in pairs:
        def enc(text):
            t = tokenizer(text, truncation=True, max_length=max_len, add_special_tokens=False)
            return torch.tensor(t["input_ids"]), torch.tensor(t["attention_mask"])
        c_text = p["prompt"] + p["chosen"] + tokenizer.eos_token
        r_text = p["prompt"] + p["rejected"] + tokenizer.eos_token
        c_ids, c_mask = enc(c_text)
        r_ids, r_mask = enc(r_text)
        prompt_len = len(tokenizer(p["prompt"], add_special_tokens=False)["input_ids"])
        c_lab = c_ids.clone(); c_lab[:prompt_len] = -100
        r_lab = r_ids.clone(); r_lab[:prompt_len] = -100
        items = [
            ("c_ids", c_ids), ("c_mask", c_mask), ("c_lab", c_lab),
            ("r_ids", r_ids), ("r_mask", r_mask), ("r_lab", r_lab),
            ("correct_idx", torch.tensor(p["correct_idx"]))
        ]
        for k, v in items:
            data[k].append(v)
    max_c = max(len(x) for x in data["c_ids"])
    max_r = max(len(x) for x in data["r_ids"])
    def pad(t, L):
        return t if len(t) >= L else torch.cat([t, torch.full((L - len(t),), pad_id, dtype=t.dtype)])
    result_tensors = []
    for k in ["c_ids","c_mask","c_lab","r_ids","r_mask","r_lab","correct_idx"]:
        if k == "correct_idx":
            result_tensors.append(torch.stack(data[k]))
        else:
            L = max_c if k.startswith("c_") else max_r
            result_tensors.append(torch.stack([pad(t, L) for t in data[k]]))
    return TensorDataset(*result_tensors)


# ==================== EVALUATION (Hindi only) ====================
@torch.no_grad()
def evaluate_hindi(model, hi_dev, tokenizer, title="Eval"):
    model.eval()
    pairs = build_dpo_pairs(hi_dev)
    if not pairs: return 0.0
    ds = tokenize_dpo(pairs, tokenizer)
    loader = DataLoader(ds, batch_size=32, shuffle=False)
    token_ids = torch.tensor([tokenizer.encode(f" {c}", add_special_tokens=False)[0] for c in "ABCD"], device=model.device)
    correct = total = 0
    for batch in loader:
        c_ids, _, c_lab, _, _, _, correct_idx = [b.to(model.device) for b in batch]
        for i in range(c_ids.size(0)):
            prompt_len = (c_lab[i] == -100).sum().item()
            inp = c_ids[i:i+1, :prompt_len]
            attn_mask = (inp != tokenizer.pad_token_id).long()
            logits = model(input_ids=inp, attention_mask=attn_mask).logits[:, -1, token_ids]
            pred = logits.argmax(-1).item()
            true = correct_idx[i].item()
            total += 1
            if pred == true:
                correct += 1
    acc = correct / total if total > 0 else 0.0
    logger.info(f"{title} → Hindi Test Accuracy: {acc:.4f} ({correct}/{total})")
    return acc

# ==================== BATCH DISTRIBUTION REPORTING ====================
@torch.no_grad()
@torch.no_grad()
def report_batch_distributions(model, batch, tokenizer, phase):
    model.eval()
    token_ids = torch.tensor([tokenizer.encode(f" {c}", add_special_tokens=False)[0] for c in "ABCD"], device=model.device)
    true_counts = Counter()
    pred_counts = Counter()
    letters = "ABCD"

    # Detect batch type by length
    batch = [b.to(model.device) for b in batch]
    if len(batch) == 7:
        # DPO batch
        c_ids, _, c_lab, _, _, _, correct_idx = batch
        input_ids_for_eval = c_ids
        labels_for_prompt_len = c_lab
    elif len(batch) == 4:
        # SFT batch: ids, mask, lab, correct_idx
        input_ids_for_eval, _, labels_for_prompt_len, correct_idx = batch
    else:
        logger.warning(f"Unexpected batch size {len(batch)} in report_batch_distributions")
        return

    for i in range(input_ids_for_eval.size(0)):
        true = letters[correct_idx[i].item()]
        true_counts[true] += 1

        # Find prompt length: where labels == -100
        prompt_len = (labels_for_prompt_len[i] == -100).sum().item()
        inp = input_ids_for_eval[i:i+1, :prompt_len]
        attn_mask = (inp != tokenizer.pad_token_id).long()

        logits = model(input_ids=inp, attention_mask=attn_mask).logits[:, -1, token_ids]
        pred_idx = logits.argmax(-1).item()
        pred = letters[pred_idx]
        pred_counts[pred] += 1

    logger.info(f"{phase} - True letter distribution: {dict(true_counts)}")
    logger.info(f"{phase} - Predicted letter distribution: {dict(pred_counts)}")
    
# ==================== GRADIENT HELPERS ====================
def flatten_grads(params):
    return torch.cat([p.grad.view(-1) for p in params if p.requires_grad])

def get_logps(logits, labels):
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), reduction='none', ignore_index=-100)
    loss = loss.view(shift_labels.shape[0], -1)
    mask = (shift_labels != -100).float()
    return -(loss * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)

# ==================== DP TRAINING FUNCTIONS ====================
def train_sft_dp(model, train_ds, tokenizer, hi_dev, name="SFT-DP"):
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)
    loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    tparams = [p for p in model.parameters() if p.requires_grad]
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for bi, batch in enumerate(loader):
            ids, mask, lab, _ = [x.to(device) for x in batch]
            per_sample_grads = []
            for i in range(ids.size(0)):
                model.zero_grad()
                loss = model(input_ids=ids[i:i+1], attention_mask=mask[i:i+1], labels=lab[i:i+1]).loss
                loss.backward()
                g = flatten_grads(tparams)
                norm = torch.norm(g)
                if norm > 0:
                    g = g * min(1.0, MAX_GRAD_NORM / norm)
                per_sample_grads.append(g)
                model.zero_grad()
            if per_sample_grads:
                g_avg = torch.stack(per_sample_grads).mean(0)
                noise = torch.normal(0, SIGMA * MAX_GRAD_NORM, size=g_avg.shape, device=device)
                g_noisy = g_avg + noise
                idx = 0
                for p in tparams:
                    n = p.numel()
                    p.grad = g_noisy[idx:idx+n].reshape(p.shape)
                    idx += n
                optimizer.step()
                optimizer.zero_grad()
            if bi == len(loader) - 1:  # Report on last batch of epoch
                report_batch_distributions(model, batch, tokenizer, f"{name} Epoch {epoch}")
        acc = evaluate_hindi(model, hi_dev, tokenizer, f"{name} After Epoch {epoch}")
    return model

def train_dpo_dp(model, ref_model, train_ds, tokenizer, hi_dev, name="DPO-DP"):
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)
    loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    tparams = [p for p in model.parameters() if p.requires_grad]
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for bi, batch in enumerate(loader):
            c_ids, c_mask, c_lab, r_ids, r_mask, r_lab, _ = [x.to(device) for x in batch]
            per_sample_grads = []
            for i in range(c_ids.size(0)):
                model.zero_grad()
                pc = get_logps(model(c_ids[i:i+1], c_mask[i:i+1]).logits, c_lab[i:i+1])
                pr = get_logps(model(r_ids[i:i+1], r_mask[i:i+1]).logits, r_lab[i:i+1])
                with torch.no_grad():
                    rc = get_logps(ref_model(c_ids[i:i+1], c_mask[i:i+1]).logits, c_lab[i:i+1])
                    rr = get_logps(ref_model(r_ids[i:i+1], r_mask[i:i+1]).logits, r_lab[i:i+1])
                loss = -F.logsigmoid(0.1 * ((pc - pr) - (rc - rr))).mean()
                loss.backward()
                g = flatten_grads(tparams)
                norm = torch.norm(g)
                if norm > 0:
                    g = g * min(1.0, MAX_GRAD_NORM / norm)
                per_sample_grads.append(g)
                model.zero_grad()
            if per_sample_grads:
                g_avg = torch.stack(per_sample_grads).mean(0)
                noise = torch.normal(0, SIGMA * MAX_GRAD_NORM, size=g_avg.shape, device=device)
                g_noisy = g_avg + noise
                idx = 0
                for p in tparams:
                    n = p.numel()
                    p.grad = g_noisy[idx:idx+n].reshape(p.shape)
                    idx += n
                optimizer.step()
                optimizer.zero_grad()
            if bi == len(loader) - 1:
                report_batch_distributions(model, batch, tokenizer, f"{name} Epoch {epoch}")
        acc = evaluate_hindi(model, hi_dev, tokenizer, f"{name} After Epoch {epoch}")
    return model

# ==================== PDP TRAINING FUNCTIONS (with projection) ====================
def train_sft_pdp(model, train_ds, tokenizer, hi_dev, V_k, name="SFT-PDP"):
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)
    loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    tparams = [p for p in model.parameters() if p.requires_grad]
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for bi, batch in enumerate(loader):
            ids, mask, lab, _ = [x.to(device) for x in batch]
            per_sample_grads = []
            for i in range(ids.size(0)):
                model.zero_grad()
                loss = model(input_ids=ids[i:i+1], attention_mask=mask[i:i+1], labels=lab[i:i+1]).loss
                loss.backward()
                g = flatten_grads(tparams)
                norm = torch.norm(g)
                if norm > 0:
                    g = g * min(1.0, MAX_GRAD_NORM / norm)
                per_sample_grads.append(g)
                model.zero_grad()
            if per_sample_grads:
                g_avg = torch.stack(per_sample_grads).mean(0)
                noise = torch.normal(0, SIGMA * MAX_GRAD_NORM, size=g_avg.shape, device=device)
                g_noisy = g_avg + noise
                g_proj = V_k @ (V_k.T @ g_noisy)  # Projection onto public subspace
                idx = 0
                for p in tparams:
                    n = p.numel()
                    p.grad = g_proj[idx:idx+n].reshape(p.shape)
                    idx += n
                optimizer.step()
                optimizer.zero_grad()
            if bi == len(loader) - 1:
                report_batch_distributions(model, batch, tokenizer, f"{name} Epoch {epoch}")
        acc = evaluate_hindi(model, hi_dev, tokenizer, f"{name} After Epoch {epoch}")
    return model

def train_dpo_pdp(model, ref_model, train_ds, tokenizer, hi_dev, V_k, name="DPO-PDP"):
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)
    loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    tparams = [p for p in model.parameters() if p.requires_grad]
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for bi, batch in enumerate(loader):
            c_ids, c_mask, c_lab, r_ids, r_mask, r_lab, _ = [x.to(device) for x in batch]
            per_sample_grads = []
            for i in range(c_ids.size(0)):
                model.zero_grad()
                pc = get_logps(model(c_ids[i:i+1], c_mask[i:i+1]).logits, c_lab[i:i+1])
                pr = get_logps(model(r_ids[i:i+1], r_mask[i:i+1]).logits, r_lab[i:i+1])
                with torch.no_grad():
                    rc = get_logps(ref_model(c_ids[i:i+1], c_mask[i:i+1]).logits, c_lab[i:i+1])
                    rr = get_logps(ref_model(r_ids[i:i+1], r_mask[i:i+1]).logits, r_lab[i:i+1])
                loss = -F.logsigmoid(0.1 * ((pc - pr) - (rc - rr))).mean()
                loss.backward()
                g = flatten_grads(tparams)
                norm = torch.norm(g)
                if norm > 0:
                    g = g * min(1.0, MAX_GRAD_NORM / norm)
                per_sample_grads.append(g)
                model.zero_grad()
            if per_sample_grads:
                g_avg = torch.stack(per_sample_grads).mean(0)
                noise = torch.normal(0, SIGMA * MAX_GRAD_NORM, size=g_avg.shape, device=device)
                g_noisy = g_avg + noise
                g_proj = V_k @ (V_k.T @ g_noisy)
                idx = 0
                for p in tparams:
                    n = p.numel()
                    p.grad = g_proj[idx:idx+n].reshape(p.shape)
                    idx += n
                optimizer.step()
                optimizer.zero_grad()
            if bi == len(loader) - 1:
                report_batch_distributions(model, batch, tokenizer, f"{name} Epoch {epoch}")
        acc = evaluate_hindi(model, hi_dev, tokenizer, f"{name} After Epoch {epoch}")
    return model

# ==================== SUBSPACE ESTIMATION FROM PUBLIC ENGLISH DATA ====================
def compute_subspace_from_public(model, public_ds, tokenizer):
    logger.info(f"Estimating subspace (k={SUBSPACE_K}) from {len(public_ds)} English examples...")
    model.eval()
    grads = []
    loader = DataLoader(public_ds, batch_size=1, shuffle=False)
    for batch in loader:
        c_ids, c_mask, c_lab, r_ids, r_mask, r_lab, _ = [x.to(device) for x in batch]
        model.zero_grad()
        pc = get_logps(model(c_ids, c_mask).logits, c_lab)
        pr = get_logps(model(r_ids, r_mask).logits, r_lab)
        loss = -F.logsigmoid(0.1 * (pc - pr)).mean()
        loss.backward()
        g = flatten_grads([p for p in model.parameters() if p.requires_grad])
        grads.append(g.cpu())
    G = torch.stack(grads).numpy()
    _, _, Vt = randomized_svd(G, n_components=SUBSPACE_K, random_state=42)
    V_k = torch.tensor(Vt.T, dtype=torch.float32, device=device)
    logger.info("Subspace estimation complete.")
    return V_k

# ==================== MODEL SETUP ====================
def get_lora_model():
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=DTYPE, attn_implementation="sdpa", local_files_only=True
    ).to(device)
    config = LoraConfig(
        r=64, lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type=TaskType.CAUSAL_LM, lora_dropout=0.05
    )
    return get_peft_model(model, config)

# ==================== MAIN ====================
def main():
    global device
    
    # Dynamic log filename including public subset size and batch size
    log_filename = f"logs/hindi_private_dp_vs_pdp_pub{PUBLIC_SUBSET_SIZE}_bs{BATCH_SIZE}_sigma{SIGMA}_k{SUBSPACE_K}_lr{LR}_epochs{EPOCHS}_seed{SEED}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        handlers=[logging.FileHandler(log_filename, mode="w"), logging.StreamHandler()]
    )
    logger.info(f"Logging to {log_filename}")
    logger.info(f"Config: PUBLIC_SUBSET_SIZE={PUBLIC_SUBSET_SIZE}, BATCH_SIZE={BATCH_SIZE}, "
                f"SIGMA={SIGMA}, EPOCHS={EPOCHS}, LR={LR}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token

    hi_test, hi_dev, en_test = load_hindi_data()

    # Private Hindi datasets (train)
    private_sft_pairs = build_sft_examples(hi_test)
    private_dpo_pairs = build_dpo_pairs(hi_test)
    private_sft_ds = tokenize_sft(private_sft_pairs, tokenizer)
    private_dpo_ds = tokenize_dpo(private_dpo_pairs, tokenizer)

    # Public English subsets for PDP folds
    en_indices = list(range(len(en_test["question"])))
    random.shuffle(en_indices)
    public_subsets = [en_indices[i*PUBLIC_SUBSET_SIZE:(i+1)*PUBLIC_SUBSET_SIZE] for i in range(NUM_FOLDS)]

    # Base model evaluation
    logger.info("\n" + "="*80)
    logger.info("BASE MODEL EVALUATION ON HINDI TEST")
    base_model = get_lora_model()
    base_acc = evaluate_hindi(base_model, hi_dev, tokenizer, "BASE MODEL")
    del base_model; gc.collect(); torch.cuda.empty_cache()

    # ==================== DP PATH ====================
    logger.info("\n" + "="*80)
    logger.info("STARTING DP PATH: SFT-DP → DPO-DP")
    model_dp = get_lora_model()
    model_dp = train_sft_dp(model_dp, private_sft_ds, tokenizer, hi_dev, "SFT-DP")
    sft_dp_acc = evaluate_hindi(model_dp, hi_dev, tokenizer, "FINAL SFT-DP")
    
    ref_dp = get_lora_model()
    ref_dp.load_state_dict(model_dp.state_dict())
    ref_dp.eval()
    
    model_dp = train_dpo_dp(model_dp, ref_dp, private_dpo_ds, tokenizer, hi_dev, "DPO-DP")
    dpo_dp_acc = evaluate_hindi(model_dp, hi_dev, tokenizer, "FINAL DPO-DP")
    
    del model_dp, ref_dp; gc.collect(); torch.cuda.empty_cache()

    # ==================== PDP PATH ====================
    logger.info("\n" + "="*80)
    logger.info(f"STARTING PDP PATH ({NUM_FOLDS} folds, public subset size={PUBLIC_SUBSET_SIZE})")
    pdp_sft_accs = []
    pdp_dpo_accs = []
    
    for fold in range(1, NUM_FOLDS + 1):
        logger.info(f"\n--- PDP Fold {fold}/{NUM_FOLDS} ---")
        idxs = public_subsets[fold-1]
        pub_dict = {k: [v[i] for i in idxs] for k, v in en_test.items()}
        pub_dpo_pairs = build_dpo_pairs(pub_dict)
        pub_dpo_ds = tokenize_dpo(pub_dpo_pairs, tokenizer)
        
        temp_model = get_lora_model()
        V_k = compute_subspace_from_public(temp_model, pub_dpo_ds, tokenizer)
        del temp_model; gc.collect(); torch.cuda.empty_cache()
        
        model_pdp = get_lora_model()
        model_pdp = train_sft_pdp(model_pdp, private_sft_ds, tokenizer, hi_dev, V_k, f"PDP-SFT Fold {fold}")
        sft_acc = evaluate_hindi(model_pdp, hi_dev, tokenizer, f"PDP-SFT Fold {fold} Final")
        pdp_sft_accs.append(sft_acc)
        
        ref_pdp = get_lora_model()
        ref_pdp.load_state_dict(model_pdp.state_dict())
        ref_pdp.eval()
        
        model_pdp = train_dpo_pdp(model_pdp, ref_pdp, private_dpo_ds, tokenizer, hi_dev, V_k, f"PDP-DPO Fold {fold}")
        dpo_acc = evaluate_hindi(model_pdp, hi_dev, tokenizer, f"PDP-DPO Fold {fold} Final")
        pdp_dpo_accs.append(dpo_acc)
        
        del model_pdp, ref_pdp, V_k; gc.collect(); torch.cuda.empty_cache()
    
    avg_pdp_sft = sum(pdp_sft_accs) / NUM_FOLDS
    avg_pdp_dpo = sum(pdp_dpo_accs) / NUM_FOLDS

    # ==================== FINAL SUMMARY ====================
    logger.info("\n" + "="*80)
    logger.info("FINAL RESULTS SUMMARY (Hindi Test Accuracy)")
    logger.info("="*80)
    logger.info(f"{'Stage':<20} {'DP':<12} {'PDP (avg over folds)':<20}")
    logger.info("-"*60)
    logger.info(f"{'Base Model':<20} {base_acc:<12.4f}")
    logger.info(f"{'After SFT':<20} {sft_dp_acc:<12.4f} {avg_pdp_sft:<20.4f}")
    logger.info(f"{'After DPO':<20} {dpo_dp_acc:<12.4f} {avg_pdp_dpo:<20.4f}")
    logger.info("="*80)

if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    main()
