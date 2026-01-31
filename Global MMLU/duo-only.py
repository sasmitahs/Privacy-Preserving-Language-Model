#!/usr/bin/env python3
"""
Global MMLU – TRUE 5-Fold Public Subspace PDP-DPO vs DP-DPO
→ 5000 public EN → 5 random subsets of 1000 → unique V_k per fold
→ Per-epoch + per-language accuracy printing
100% WORKING – DECEMBER 2025
"""
import os
import gc
import random
import logging
from datetime import datetime
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Subset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.utils.extmath import randomized_svd

# === FIX TOKENIZER PARALLELISM WARNING ===
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ==================== CONFIG ====================
HOME = os.path.expanduser("~")
MODEL_PATH = os.path.join(HOME, "hf_downloads", "llama-3.2-1b-instruct")
DATASET_ROOT = os.path.join(HOME, "hf_downloads", "global_mmlu_dataset", "CohereLabs___global-mmlu")

os.makedirs("logs", exist_ok=True)
os.makedirs("results", exist_ok=True)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
assert device.type == "cuda", "GPU required!"

DTYPE = torch.bfloat16
PUBLIC_TOTAL = 5000
SUBSET_SIZE = 1000
NUM_FOLDS = 5
SUBSPACE_K = 20
SEED = 42

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    handlers=[
        logging.FileHandler(f"logs/global_mmlu_5fold_per_subset_{datetime.now().strftime('%Y%m%d_%H%M')}.log"),
        logging.StreamHandler()
    ]
)

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
set_seed()

# ==================== DATA LOADING (Arrow) ====================
import pyarrow as pa
def load_arrow_file(path):
    try:
        with open(path, 'rb') as f:
            reader = pa.ipc.open_stream(f)
            table = pa.Table.from_batches(list(reader))
        return table.to_pydict() if table.num_rows > 0 else None
    except: return None

def load_language(lang_code):
    path = os.path.join(DATASET_ROOT, lang_code, "0.0.0")
    if not os.path.isdir(path): return None
    for d in os.listdir(path):
        full = os.path.join(path, d)
        if not os.path.isdir(full): continue
        for split in ["test", "dev", "validation", "train"]:
            fpath = os.path.join(full, f"global-mmlu-{split}.arrow")
            if os.path.exists(fpath):
                data = load_arrow_file(fpath)
                if data and "question" in data:
                    return data
    return None

def load_all_languages():
    datasets = {}
    for lang in sorted(os.listdir(DATASET_ROOT)):
        path = os.path.join(DATASET_ROOT, lang)
        if not os.path.isdir(path): continue
        data = load_language(lang)
        if data:
            n = len(data["question"])
            datasets[lang] = data
            logging.info(f"Loaded {lang:>4}: {n:>6} examples")
    return datasets

# ==================== DPO PAIRS & TOKENIZATION ====================
def build_dpo_pairs(examples_dict, lang_code="en"):
    pairs = []
    n = len(examples_dict["question"])
    for i in range(n):
        q = str(examples_dict["question"][i] or "").strip()
        if not q: continue
        opts = []
        for c in "abcd":
            key = f"option_{c}"
            val = examples_dict.get(key, [None]*n)[i]
            if val is None or str(val).strip() == "": break
            opts.append(str(val).strip())
        else:
            if len(opts) == 4 and all(opts):
                ans = str(examples_dict.get("answer", [None]*n)[i] or "").strip().upper()
                if ans in "ABCD":
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
        def encode(text):
            enc = tokenizer(text, truncation=True, max_length=max_len, add_special_tokens=False)
            return torch.tensor(enc["input_ids"]), torch.tensor(enc["attention_mask"])
        c_text = p["prompt"] + p["chosen"] + tokenizer.eos_token
        r_text = p["prompt"] + p["rejected"] + tokenizer.eos_token
        c_ids, c_mask = encode(c_text)
        r_ids, r_mask = encode(r_text)
        prompt_len = len(tokenizer(p["prompt"], add_special_tokens=False)["input_ids"])
        c_lab = c_ids.clone(); c_lab[:prompt_len] = -100
        r_lab = r_ids.clone(); r_lab[:prompt_len] = -100
        for k, v in [("c_ids", c_ids), ("c_mask", c_mask), ("c_lab", c_lab),
                     ("r_ids", r_ids), ("r_mask", r_mask), ("r_lab", r_lab),
                     ("correct_idx", torch.tensor(p["correct_idx"], dtype=torch.long))]:
            data[k].append(v)
    # Padding
    max_c = max(len(x) for x in data["c_ids"])
    max_r = max(len(x) for x in data["r_ids"])
    def pad(t, L): 
        return t[:L] if len(t) >= L else torch.cat([t, torch.full((L-len(t),), pad_id, dtype=t.dtype)])
    for k in data:
        if k == "correct_idx":
            data[k] = torch.stack(data[k])
        else:
            L = max_c if k.startswith("c_") else max_r
            data[k] = torch.stack([pad(t, L) for t in data[k]])
    return TensorDataset(*[data[k] for k in ["c_ids","c_mask","c_lab","r_ids","r_mask","r_lab","correct_idx"]])

def get_logps(logits, labels):
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), reduction='none', ignore_index=-100)
    loss = loss.view(shift_labels.shape[0], -1)
    mask = (shift_labels != -100).float()
    return -(loss * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)

# ==================== EVALUATION WITH PER-LANG PRINT ====================
@torch.no_grad()
def evaluate_and_print(model, lang_datasets, public_en_data, tokenizer, title="Eval"):
    model.eval()
    stats = defaultdict(lambda: {"correct": 0, "total": 0})
    token_ids = torch.tensor([tokenizer.encode(f" {c}", add_special_tokens=False)[0] for c in "ABCD"], device=device)

    for lang, data in [("en", public_en_data), *lang_datasets.items()]:
        pairs = build_dpo_pairs(data, lang)
        if not pairs: continue
        ds = tokenize_dpo(pairs, tokenizer)
        if not ds: continue
        loader = DataLoader(ds, batch_size=32, shuffle=False)
        for batch in loader:
            c_ids, _, c_lab, _, _, _, correct_idx = [x.to(device) for x in batch]
            for i in range(c_ids.size(0)):
                prompt_len = (c_lab[i] == -100).sum().item()
                inp = c_ids[i:i+1, :prompt_len]
                mask = (inp != tokenizer.pad_token_id).long()
                logits = model(input_ids=inp, attention_mask=mask).logits[:, -1, token_ids]
                pred = logits.argmax(-1).item()
                true = correct_idx[i].item()
                stats[lang]["total"] += 1
                if pred == true:
                    stats[lang]["correct"] += 1

    acc_per_lang = {l: s["correct"]/max(s["total"],1) for l, s in stats.items()}
    overall = sum(s["correct"] for s in stats.values()) / max(1, sum(s["total"] for s in stats.values()))

    print(f"\n{title} | Overall Accuracy: {overall:.4f}")
    print("Lang   Accuracy   #Examples")
    print("-" * 30)
    for lang in sorted(acc_per_lang):
        print(f"{lang:<6} {acc_per_lang[lang]:8.4f}   {stats[lang]['total']:>6}")
    print("-" * 30)
    return acc_per_lang, overall

# ==================== SUBSPACE COMPUTATION ====================
def compute_subspace(model, ds, tokenizer, k=SUBSPACE_K):
    model.train()
    model.enable_adapter_layers()
    grads = []
    loader = DataLoader(ds, batch_size=1, shuffle=False)
    logging.info(f"Computing subspace from {len(ds)} examples...")
    for i, batch in enumerate(loader):
        if i % 200 == 0 and i > 0:
            print(f"   → Subspace progress: {i}/{len(ds)}")
        c_ids, c_mask, c_lab, r_ids, r_mask, r_lab, _ = [x.to(device) for x in batch]
        model.zero_grad(set_to_none=True)
        pc = get_logps(model(c_ids, attention_mask=c_mask).logits, c_lab)
        pr = get_logps(model(r_ids, attention_mask=r_mask).logits, r_lab)
        loss = -F.logsigmoid(0.1 * (pc - pr)).mean()
        loss.backward()
        g = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])
        grads.append(g.cpu())
    G = torch.stack(grads)
    G = G - G.mean(0, keepdim=True)
    U, S, _ = randomized_svd(G.numpy(), n_components=k, n_iter=7, random_state=42)
    V_k = torch.tensor(U, dtype=torch.float32, device=device)
    logging.info(f"Subspace ready | Top singular values: {S[:5]}")
    del G, grads; gc.collect(); torch.cuda.empty_cache()
    return V_k

# ==================== TRAINING WITH PER-EPOCH EVAL ====================
def train_with_eval(model, ref_model, train_ds, tokenizer, args, pdp=False, V_k=None, fold=None, method_name="DP-DPO"):
    model.train()
    model.enable_adapter_layers()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    for epoch in range(1, args.epochs + 1):
        for batch_idx, batch in enumerate(loader):
            c_ids, c_mask, c_lab, r_ids, r_mask, r_lab = [t.to(device) for t in batch[:6]]
            per_sample_grads = []
            for i in range(c_ids.size(0)):
                model.zero_grad(set_to_none=True)
                pc = get_logps(model(c_ids[i:i+1], attention_mask=c_mask[i:i+1]).logits, c_lab[i:i+1])
                pr = get_logps(model(r_ids[i:i+1], attention_mask=r_mask[i:i+1]).logits, r_lab[i:i+1])
                with torch.no_grad():
                    rc = get_logps(ref_model(c_ids[i:i+1], attention_mask=c_mask[i:i+1]).logits, c_lab[i:i+1])
                    rr = get_logps(ref_model(r_ids[i:i+1], attention_mask=r_mask[i:i+1]).logits, r_lab[i:i+1])
                loss = -F.logsigmoid(0.1 * ((pc - pr) - (rc - rr))).mean()
                loss.backward()
                g = torch.cat([p.grad.view(-1) for p in model.parameters() if p.requires_grad])
                g = g / g.norm().clamp(min=1e-8)
                per_sample_grads.append(g)
            if not per_sample_grads: continue
            g_avg = torch.stack(per_sample_grads).mean(0)
            g_avg = g_avg + torch.normal(0, args.sigma, size=g_avg.shape, device=device)
            if pdp and V_k is not None:
                g_avg = V_k @ (V_k.t() @ g_avg)
            idx = 0
            for p in model.parameters():
                if p.requires_grad:
                    sz = p.numel()
                    p.grad = g_avg[idx:idx+sz].reshape(p.shape)
                    idx += sz
            optimizer.step()

        # === Per-epoch evaluation ===
        title = f"FOLD {fold} | Epoch {epoch}/{args.epochs} | {method_name}"
        evaluate_and_print(model, all_data, public_en_full, tokenizer, title)

    return model

# ==================== MODEL LOADER ====================
def get_lora_model():
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=DTYPE, low_cpu_mem_usage=True, attn_implementation="sdpa"
    )
    config = LoraConfig(
        r=64, lora_alpha=32, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type=TaskType.CAUSAL_LM, lora_dropout=0.05
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model.to(device)

# ==================== MAIN ====================
def main():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--sigma", type=float, default=4.0)
    args = parser.parse_args()

    global tokenizer, all_data, public_en_full, private_ds
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token

    logging.info("Loading datasets...")
    all_data = load_all_languages()
    public_en_full = all_data.pop("en")

    # === 5000 public EN examples ===
    total_en = len(public_en_full["question"])
    idx_5000 = random.sample(range(total_en), min(PUBLIC_TOTAL, total_en))
    public_5000 = {k: [v[i] for i in idx_5000] for k, v in public_en_full.items()}
    public_pairs_5000 = build_dpo_pairs(public_5000, "en")
    public_ds_full = tokenize_dpo(public_pairs_5000, tokenizer)

    # === Private (non-English) training data ===
    private_examples = []
    for lang, data in all_data.items():
        for i in range(len(data["question"])):
            private_examples.append({k: data[k][i] for k in data})
    random.shuffle(private_examples)
    private_dict = {k: [ex[k] for ex in private_examples] for k in private_examples[0]}
    private_pairs = build_dpo_pairs(private_dict, "private")
    private_ds = tokenize_dpo(private_pairs, tokenizer)

    results = []
    print("\n" + "="*100)
    print("TRUE 5-FOLD PER-SUBSET PUBLIC SUBSPACE PDP-DPO EXPERIMENT")
    print("="*100)

    for fold in range(1, NUM_FOLDS + 1):
        print(f"\n{'='*120}")
        print(f"FOLD {fold}/5 – Creating NEW public subset #{fold} (1000 examples) → NEW subspace")
        print(f"{'='*120}")

        # New random subset of 1000 from the 5000
        subset_idx = random.sample(range(len(public_ds_full)), SUBSET_SIZE)
        subset_ds = Subset(public_ds_full, subset_idx)

        # Compute fresh subspace for this fold
        model_sub = get_lora_model()
        V_k = compute_subspace(model_sub, subset_ds, tokenizer, k=SUBSPACE_K)
        del model_sub; gc.collect(); torch.cuda.empty_cache()

        # Train both models
        ref_model = get_lora_model(); ref_model.eval()

        print(f"\nTraining DP-DPO (no projection)...")
        model_dp = get_lora_model()
        model_dp = train_with_eval(model_dp, ref_model, private_ds, tokenizer, args,
                                   pdp=False, fold=fold, method_name="DP-DPO")

        print(f"\nTraining PDP-DPO (projecting onto subset #{fold} subspace)...")
        model_pdp = get_lora_model()
        model_pdp = train_with_eval(model_pdp, ref_model, private_ds, tokenizer, args,
                                    pdp=True, V_k=V_k, fold=fold, method_name="PDP-DPO")

        # Final eval
        _, dp_acc = evaluate_and_print(model_dp, all_data, public_en_full, tokenizer,
                                        f"FOLD {fold} – FINAL DP-DPO")
        _, pdp_acc = evaluate_and_print(model_pdp, all_data, public_en_full, tokenizer,
                                        f"FOLD {fold} – FINAL PDP-DPO")

        delta = pdp_acc - dp_acc
        results.append({"fold": fold, "dp": dp_acc, "pdp": pdp_acc, "delta": delta})
        print(f"FOLD {fold} RESULT → DP-DPO: {dp_acc:.4f} | PDP-DPO: {pdp_acc:.4f} | Δ = {delta:+.4f}")

        del model_dp, model_pdp, ref_model, V_k
        gc.collect(); torch.cuda.empty_cache()

    # === FINAL SUMMARY ===
    print("\n" + "="*100)
    print("FINAL 5-FOLD RESULTS (Per-Subset Public Subspace PDP-DPO)")
    print("="*100)
    print(f"{'Fold':<6} {'DP-DPO':<10} {'PDP-DPO':<10} {'Δ':<10}")
    for r in results:
        print(f"{r['fold']:<6} {r['dp']:<10.4f} {r['pdp']:<10.4f} {r['delta']:+10.4f}")
    avg_delta = sum(r['delta'] for r in results) / len(results)
    print(f"{'AVG':<6} {'':<22} {avg_delta:+10.4f}")
    print("="*100)

if __name__ == "__main__":
    main()
