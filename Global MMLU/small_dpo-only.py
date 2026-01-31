#!/usr/bin/env python3
"""
Global-MMLU Small Experiment – FIXED VERSION
→ Public: 3000 EN test → 5 × 600 disjoint
→ Private: 150 per non-EN from test
→ Evaluation: ONLY dev split
→ Final table: Baseline + each of 5 PDP folds separately
"""

import os
import gc
import random
import logging
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Subset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.utils.extmath import randomized_svd

# ==================== CONFIG ====================
HOME = os.path.expanduser("~")
MODEL_PATH = os.path.join(HOME, "hf_downloads", "llama-3.2-1b-instruct")
DATASET_ROOT = os.path.join(HOME, "hf_downloads", "global_mmlu_dataset", "CohereLabs___global-mmlu")

os.makedirs("logs", exist_ok=True)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
assert device.type == "cuda", "GPU required!"

DTYPE = torch.bfloat16
PUBLIC_TOTAL = 10000
SUBSET_SIZE = 2000
NUM_FOLDS = 5
SUBSPACE_K = 20
PRIVATE_PER_LANG = 14000
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    handlers=[
        logging.FileHandler("logs/language_small_epoch3_14000private.log", mode="w"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ==================== DATA LOADING ====================
import pyarrow as pa

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

def load_all_languages():
    data = {}
    for lang in sorted(os.listdir(DATASET_ROOT)):
        path = os.path.join(DATASET_ROOT, lang)
        if not os.path.isdir(path): continue
        test_data = load_split(lang, "test")
        dev_data = load_split(lang, "dev")
        if test_data and dev_data:
            data[lang] = {"test": test_data, "dev": dev_data}
            nt = len(test_data["question"])
            nd = len(dev_data["question"])
            logger.info(f"Loaded {lang:>4} | test: {nt:>5} | dev: {nd:>5}")
    return data

# ==================== DPO PAIRS & TOKENIZATION ====================
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
        wrong_idx = (correct_idx + 1 + random.randint(0, 2)) % 4
        prompt = f"Question: {q}\n" + "\n".join(f"{chr(65+j)}. {opts[j]}" for j in range(4)) + "\nAnswer:"
        pairs.append({
            "prompt": prompt,
            "chosen": f" {ans}",
            "rejected": f" {chr(65 + wrong_idx)}",
            "correct_idx": correct_idx
        })
    return pairs

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

    # Padding
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

def get_logps(logits, labels):
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), reduction='none', ignore_index=-100)
    loss = loss.view(shift_labels.shape[0], -1)
    mask = (shift_labels != -100).float()
    return -(loss * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)

# ==================== EVALUATION ON DEV ====================
@torch.no_grad()
def evaluate_dev(model, all_data, tokenizer, title="Eval"):
    model.eval()
    stats = defaultdict(lambda: {"correct": 0, "total": 0})
    token_ids = torch.tensor([tokenizer.encode(f" {c}", add_special_tokens=False)[0] for c in "ABCD"], device=device)

    for lang, splits in all_data.items():
        pairs = build_dpo_pairs(splits["dev"])
        if not pairs: continue
        ds = tokenize_dpo(pairs, tokenizer)
        if not ds: continue
        loader = DataLoader(ds, batch_size=32, shuffle=False)
        for batch in loader:
            c_ids, _, c_lab, _, _, _, correct_idx = [b.to(device) for b in batch]
            for i in range(c_ids.size(0)):
                prompt_len = (c_lab[i] == -100).sum().item()
                inp = c_ids[i:i+1, :prompt_len]
                attn_mask = (inp != tokenizer.pad_token_id).long()
                logits = model(input_ids=inp, attention_mask=attn_mask).logits[:, -1, token_ids]
                pred = logits.argmax(-1).item()
                true = correct_idx[i].item()
                stats[lang]["total"] += 1
                if pred == true:
                    stats[lang]["correct"] += 1

    accs = {l: s["correct"] / max(1, s["total"]) for l, s in stats.items()}
    overall = sum(s["correct"] for s in stats.values()) / max(1, sum(s["total"] for s in stats.values()))
    logger.info(f"\n{title} → Overall Dev Accuracy: {overall:.4f}")
    for l in sorted(accs):
        logger.info(f"  {l:<4}: {accs[l]:.4f} ({stats[l]['total']} examples)")
    return overall, accs

# ==================== MODEL HELPERS ====================
def get_lora_model():
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=DTYPE, low_cpu_mem_usage=True, attn_implementation="sdpa"
    ).to(device)
    config = LoraConfig(
        r=64, lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type=TaskType.CAUSAL_LM, lora_dropout=0.05
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model

def compute_subspace(model, ds, tokenizer):
    """
    Compute top-k subspace from gradient matrix.
    FIXED: Returns V_k with shape (k, num_params) for proper projection.
    """
    logger.info(f"Computing subspace (k={SUBSPACE_K}) from {len(ds)} examples...")
    model.train()
    grads = []
    loader = DataLoader(ds, batch_size=1, shuffle=False)
    
    for i, batch in enumerate(loader):
        if i % 100 == 0:
            logger.info(f"   → {i}/{len(ds)}")
        c_ids, c_mask, c_lab, r_ids, r_mask, r_lab, _ = [x.to(device) for x in batch]
        model.zero_grad()
        pc = get_logps(model(c_ids, attention_mask=c_mask).logits, c_lab)
        pr = get_logps(model(r_ids, attention_mask=r_mask).logits, r_lab)
        loss = -F.logsigmoid(0.1 * (pc - pr)).mean()
        loss.backward()
        g = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])
        grads.append(g.cpu())
    
    # Stack gradients: shape (num_samples, num_params)
    G = torch.stack(grads)
    G = G - G.mean(0, keepdim=True)
    
    # SVD: U has shape (num_samples, k), Vt has shape (k, num_params)
    U, S, Vt = randomized_svd(G.numpy(), n_components=SUBSPACE_K, n_iter=7, random_state=42)
    
    # V_k should be (k, num_params) - this is Vt from SVD
    V_k = torch.tensor(Vt, dtype=torch.float32, device=device)
    
    logger.info(f"Subspace ready. Shape: {V_k.shape}")
    logger.info(f"Top singular values: {S[:5]}")
    return V_k

def train_one_model(ref_model, train_ds, tokenizer, args, pdp=False, V_k=None, name="Model"):
    logger.info(f"\n=== TRAINING {name} | PDP={'ON' if pdp else 'OFF'} ===")
    model = get_lora_model()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    for epoch in range(1, args.epochs + 1):
        logger.info(f"\nEpoch {epoch}/{args.epochs}")
        for bi, batch in enumerate(loader):
            if bi % 20 == 0:
                logger.info(f"  batch {bi}/{len(loader)}")
            c_ids, c_mask, c_lab, r_ids, r_mask, r_lab = [x.to(device) for x in batch[:6]]
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
                g = torch.cat([p.grad.view(-1) for p in model.parameters() if p.requires_grad])
                g = g / g.norm().clamp(min=1e-8)
                per_sample_grads.append(g)
            if not per_sample_grads: continue
            g_avg = torch.stack(per_sample_grads).mean(0)
            g_avg += torch.normal(0, args.sigma, size=g_avg.shape, device=device)
            
            # FIXED: Proper projection with V_k of shape (k, num_params)
            if pdp and V_k is not None:
                # Project: g_proj = V_k^T @ (V_k @ g_avg)
                # V_k: (k, d), g_avg: (d,)
                # V_k @ g_avg: (k,)
                # V_k^T @ result: (d,)
                g_avg = V_k.t() @ (V_k @ g_avg)

            idx = 0
            for p in model.parameters():
                if p.requires_grad:
                    n = p.numel()
                    p.grad = g_avg[idx:idx+n].reshape(p.shape)
                    idx += n
            optimizer.step()

        evaluate_dev(model, all_data, tokenizer, f"{name} – Epoch {epoch}")

    return model

# ==================== MAIN ====================
def main():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--sigma", type=float, default=4.0)
    args = parser.parse_args()

    global all_data, tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loading datasets...")
    all_data = load_all_languages()

    # Public EN: 3000 from test
    en_test = all_data["en"]["test"]
    idxs = random.sample(range(len(en_test["question"])), PUBLIC_TOTAL)
    public_dict = {k: [v[i] for i in idxs] for k, v in en_test.items()}
    public_pairs = build_dpo_pairs(public_dict)
    public_ds = tokenize_dpo(public_pairs, tokenizer)
    logger.info(f"Public EN: {len(public_ds)} DPO pairs")

    # Private: 150 per non-EN
    private_examples = []
    for lang, splits in all_data.items():
        if lang == "en": continue
        test = splits["test"]
        n = len(test["question"])
        chosen = random.sample(range(n), min(PRIVATE_PER_LANG, n))
        for i in chosen:
            private_examples.append({k: test[k][i] for k in test})
    private_dict = {k: [ex[k] for ex in private_examples] for k in private_examples[0]} if private_examples else {}
    private_pairs = build_dpo_pairs(private_dict)
    private_ds = tokenize_dpo(private_pairs, tokenizer)
    logger.info(f"Private: {len(private_ds)} DPO pairs")

    # 5-fold split
    indices = list(range(len(public_ds)))
    random.shuffle(indices)
    subsets = [Subset(public_ds, indices[i*SUBSET_SIZE:(i+1)*SUBSET_SIZE]) for i in range(NUM_FOLDS)]

    # Result storage
    table = defaultdict(list)  # lang → [baseline_acc, fold1_acc, fold2_acc, ...]
    overall_scores = []

    ref_model = get_lora_model()
    ref_model.eval()

    # === Baseline ===
    logger.info("\n" + "="*100)
    logger.info("TRAINING BASELINE (DP-DPO)")
    baseline_model = train_one_model(ref_model, private_ds, tokenizer, args, pdp=False, name="Baseline")
    _, accs = evaluate_dev(baseline_model, all_data, tokenizer, "FINAL BASELINE")
    for lang, acc in accs.items():
        table[lang].append(acc)
    overall_scores.append(sum(accs.values()) / len(accs))
    del baseline_model; gc.collect(); torch.cuda.empty_cache()

    # === 5 PDP Folds ===
    for fold in range(1, 6):
        logger.info(f"\nFOLD {fold}/5 – PDP-DPO")
        subset = subsets[fold-1]
        temp_model = get_lora_model()
        V_k = compute_subspace(temp_model, subset, tokenizer)
        del temp_model; gc.collect(); torch.cuda.empty_cache()

        pdp_model = train_one_model(ref_model, private_ds, tokenizer, args, pdp=True, V_k=V_k,
                                    name=f"PDP Fold {fold}")
        _, accs = evaluate_dev(pdp_model, all_data, tokenizer, f"FINAL PDP FOLD {fold}")
        for lang, acc in accs.items():
            table[lang].append(acc)
        overall_scores.append(sum(accs.values()) / len(accs))
        del pdp_model, V_k; gc.collect(); torch.cuda.empty_cache()

    # ==================== FINAL TABLE ====================
    logger.info("\n" + "="*120)
    logger.info("FINAL RESULTS – DEV SPLIT – PER FOLD")
    logger.info("="*120)
    header = f"{'Lang':<6} {'Baseline':>10} {'Fold-1':>10} {'Fold-2':>10} {'Fold-3':>10} {'Fold-4':>10} {'Fold-5':>10}"
    logger.info(header)
    logger.info("-" * len(header))

    for lang in sorted(table):
        row = table[lang]
        while len(row) < 6:
            row.append(0.0)
        logger.info(f"{lang:<6} {row[0]:10.4f} {row[1]:10.4f} {row[2]:10.4f} {row[3]:10.4f} {row[4]:10.4f} {row[5]:10.4f}")

    logger.info("-" * len(header))
    logger.info(f"{'OVERALL':<6} {overall_scores[0]:10.4f} {overall_scores[1]:10.4f} {overall_scores[2]:10.4f} {overall_scores[3]:10.4f} {overall_scores[4]:10.4f} {overall_scores[5]:10.4f}")
    logger.info("="*120)

if __name__ == "__main__":
    main()
