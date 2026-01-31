#!/usr/bin/env python3
"""
Hindi-only training: Plain SFT + DPO
Evaluation uses DPO-style preference logic:
  "Which option is preferred relative to another option?"

NO DP
NO PDP
NO noise
NO gradient clipping
"""

import os
import gc
import random
import logging
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
import pyarrow as pa

# ==================== CONFIG ====================
HOME = os.path.expanduser("~")
MODEL_PATH = os.path.join(HOME, "llama_models", "llama-3.1-8b-instruct")
DATASET_ROOT = os.path.join(
    HOME, "hf_downloads", "global_mmlu_dataset", "CohereLabs___global-mmlu"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
assert device.type == "cuda", "GPU required!"

DTYPE = torch.bfloat16
BATCH_SIZE = 1
LR = 1e-4
EPOCHS = 3
SEED = 42
TEST_SPLIT_RATIO = 0.2
MAX_LEN = 512

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

os.makedirs("logs", exist_ok=True)

log_filename = f"logs/hindi_plain_pref_sft_dpo_bs{BATCH_SIZE}_lr{LR}_ep{EPOCHS}_seed{SEED}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    handlers=[logging.FileHandler(log_filename, mode="w"),
              logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# ==================== DATA LOADING ====================
def load_split(lang, split_name):
    path = os.path.join(DATASET_ROOT, lang, "0.0.0")
    if not os.path.isdir(path):
        return None
    for d in os.listdir(path):
        full = os.path.join(path, d)
        if not os.path.isdir(full):
            continue
        fpath = os.path.join(full, f"global-mmlu-{split_name}.arrow")
        if os.path.exists(fpath):
            with open(fpath, "rb") as f:
                reader = pa.ipc.open_stream(f)
                table = pa.Table.from_batches(list(reader))
                return table.to_pydict()
    return None


def load_and_split_hindi():
    hi_test = load_split("hi", "test")
    hi_dev = load_split("hi", "dev")
    if not (hi_test and hi_dev):
        raise ValueError("Missing Hindi test/dev data")

    combined = {k: hi_test[k] + hi_dev[k] for k in hi_test}
    n = len(combined["question"])
    idx = list(range(n))
    random.shuffle(idx)

    split = int(n * (1 - TEST_SPLIT_RATIO))
    train_idx, test_idx = idx[:split], idx[split:]

    hi_train = {k: [combined[k][i] for i in train_idx] for k in combined}
    hi_test_new = {k: [combined[k][i] for i in test_idx] for k in combined}

    logger.info(f"Hindi total: {n}")
    logger.info(f" → Train: {len(train_idx)}")
    logger.info(f" → Test : {len(test_idx)}")

    return hi_train, hi_test_new

# ==================== PAIR BUILDERS ====================
def build_sft_examples(examples):
    pairs = []
    n = len(examples["question"])
    for i in range(n):
        q = str(examples["question"][i]).strip()
        opts = []
        for c in "abcd":
            v = examples.get(f"option_{c}", [None]*n)[i]
            if not v:
                break
            opts.append(str(v).strip())
        if len(opts) != 4:
            continue
        ans = str(examples["answer"][i]).strip().upper()
        if ans not in "ABCD":
            continue
        prompt = (
            f"Question: {q}\n"
            + "\n".join(f"{chr(65+j)}. {opts[j]}" for j in range(4))
            + "\nAnswer:"
        )
        pairs.append({
            "prompt": prompt,
            "completion": f" {ans}",
            "correct_letter": ans,
        })
    return pairs


def build_dpo_pairs(examples):
    pairs = []
    n = len(examples["question"])
    for i in range(n):
        q = str(examples["question"][i]).strip()
        opts = []
        for c in "abcd":
            v = examples.get(f"option_{c}", [None]*n)[i]
            if not v:
                break
            opts.append(str(v).strip())
        if len(opts) != 4:
            continue
        ans = str(examples["answer"][i]).strip().upper()
        if ans not in "ABCD":
            continue

        correct_idx = ord(ans) - 65
        wrong_idx = random.choice([j for j in range(4) if j != correct_idx])

        prompt = (
            f"Question: {q}\n"
            + "\n".join(f"{chr(65+j)}. {opts[j]}" for j in range(4))
            + "\nAnswer:"
        )

        pairs.append({
            "prompt": prompt,
            "chosen": f" {ans}",
            "rejected": f" {chr(65 + wrong_idx)}",
            "correct_idx": correct_idx,
        })
    return pairs

# ==================== TOKENIZATION ====================
def tokenize_sft(pairs, tokenizer):
    pad = tokenizer.pad_token_id
    ids, mask, lab, cidx = [], [], [], []

    for p in pairs:
        full = p["prompt"] + p["completion"] + tokenizer.eos_token
        enc = tokenizer(full, truncation=True, max_length=MAX_LEN)
        input_ids = torch.tensor(enc["input_ids"])
        attention = torch.tensor(enc["attention_mask"])
        labels = input_ids.clone()
        plen = len(tokenizer(p["prompt"], add_special_tokens=False)["input_ids"])
        labels[:plen] = -100

        ids.append(input_ids)
        mask.append(attention)
        lab.append(labels)
        cidx.append(ord(p["correct_letter"]) - 65)

    maxlen = max(len(x) for x in ids)

    def pad_to(x):
        return torch.cat([x, torch.full((maxlen-len(x),), pad)]) if len(x) < maxlen else x

    return TensorDataset(
        torch.stack([pad_to(x) for x in ids]),
        torch.stack([pad_to(x) for x in mask]),
        torch.stack([pad_to(x) for x in lab]),
        torch.tensor(cidx),
    )


def tokenize_dpo(pairs, tokenizer):
    pad = tokenizer.pad_token_id
    data = defaultdict(list)

    for p in pairs:
        def enc(txt):
            t = tokenizer(txt, truncation=True, max_length=MAX_LEN, add_special_tokens=False)
            return torch.tensor(t["input_ids"]), torch.tensor(t["attention_mask"])

        c_ids, c_mask = enc(p["prompt"] + p["chosen"] + tokenizer.eos_token)
        r_ids, r_mask = enc(p["prompt"] + p["rejected"] + tokenizer.eos_token)

        plen = len(tokenizer(p["prompt"], add_special_tokens=False)["input_ids"])
        c_lab = c_ids.clone(); c_lab[:plen] = -100
        r_lab = r_ids.clone(); r_lab[:plen] = -100

        for k, v in [
            ("c_ids", c_ids), ("c_mask", c_mask), ("c_lab", c_lab),
            ("r_ids", r_ids), ("r_mask", r_mask), ("r_lab", r_lab),
            ("correct_idx", torch.tensor(p["correct_idx"]))
        ]:
            data[k].append(v)

    max_c = max(len(x) for x in data["c_ids"])
    max_r = max(len(x) for x in data["r_ids"])

    def pad_to(x, L):
        return torch.cat([x, torch.full((L-len(x),), pad)]) if len(x) < L else x

    tensors = []
    for k in ["c_ids","c_mask","c_lab","r_ids","r_mask","r_lab","correct_idx"]:
        if k == "correct_idx":
            tensors.append(torch.stack(data[k]))
        else:
            L = max_c if k.startswith("c_") else max_r
            tensors.append(torch.stack([pad_to(x, L) for x in data[k]]))

    return TensorDataset(*tensors)

# ==================== EVALUATION (Preference-style) ====================
@torch.no_grad()
def evaluate_hindi(model, hi_test, tokenizer, title):
    model.eval()
    pairs = build_dpo_pairs(hi_test)
    ds = tokenize_dpo(pairs, tokenizer)
    loader = DataLoader(ds, batch_size=32)

    token_ids = torch.tensor(
        [tokenizer.encode(f" {c}", add_special_tokens=False)[0] for c in "ABCD"],
        device=device,
    )

    correct = total = 0
    for batch in loader:
        c_ids, _, c_lab, _, _, _, correct_idx = [b.to(device) for b in batch]
        for i in range(c_ids.size(0)):
            plen = (c_lab[i] == -100).sum().item()
            inp = c_ids[i:i+1, :plen]
            attn = (inp != tokenizer.pad_token_id).long()
            logits = model(inp, attention_mask=attn).logits[:, -1, token_ids]
            pred = logits.argmax(-1).item()
            if pred == correct_idx[i].item():
                correct += 1
            total += 1

    acc = correct / total
    logger.info(f"{title} → Hindi Accuracy: {acc:.4f} ({correct}/{total})")
    return acc

# ==================== TRAINING ====================
def train_sft(model, ds, tokenizer, hi_test):
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

    for ep in range(1, EPOCHS+1):
        model.train()
        for ids, mask, lab, _ in loader:
            ids, mask, lab = ids.to(device), mask.to(device), lab.to(device)
            opt.zero_grad()
            loss = model(ids, attention_mask=mask, labels=lab).loss
            loss.backward()
            opt.step()
        evaluate_hindi(model, hi_test, tokenizer, f"SFT Epoch {ep}")
    return model


def get_logps(logits, labels):
    shift_logits = logits[..., :-1, :]
    shift_labels = labels[..., 1:]
    loss = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
        ignore_index=-100,
        reduction="none",
    )
    loss = loss.view(shift_labels.size(0), -1)
    mask = (shift_labels != -100).float()
    return -(loss * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)


def train_dpo(model, ref_model, ds, tokenizer, hi_test):
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

    for ep in range(1, EPOCHS+1):
        model.train()
        for c_ids, c_mask, c_lab, r_ids, r_mask, r_lab, _ in loader:
            c_ids, c_mask, c_lab = c_ids.to(device), c_mask.to(device), c_lab.to(device)
            r_ids, r_mask, r_lab = r_ids.to(device), r_mask.to(device), r_lab.to(device)

            opt.zero_grad()
            pc = get_logps(model(c_ids, c_mask).logits, c_lab)
            pr = get_logps(model(r_ids, r_mask).logits, r_lab)

            with torch.no_grad():
                rc = get_logps(ref_model(c_ids, c_mask).logits, c_lab)
                rr = get_logps(ref_model(r_ids, r_mask).logits, r_lab)

            loss = -F.logsigmoid(0.1 * ((pc - pr) - (rc - rr))).mean()
            loss.backward()
            opt.step()

        evaluate_hindi(model, hi_test, tokenizer, f"DPO Epoch {ep}")
    return model

# ==================== MODEL ====================
def get_lora_model():
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=DTYPE, attn_implementation="sdpa"
    ).to(device)
    config = LoraConfig(
        r=8,
        lora_alpha=8,
        target_modules=["q_proj","k_proj","v_proj","o_proj"],
        task_type=TaskType.CAUSAL_LM,
        lora_dropout=0.05,
    )
    return get_peft_model(base, config)

# ==================== MAIN ====================
def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token

    hi_train, hi_test = load_and_split_hindi()

    sft_ds = tokenize_sft(build_sft_examples(hi_train), tokenizer)
    dpo_ds = tokenize_dpo(build_dpo_pairs(hi_train), tokenizer)

    logger.info("BASE MODEL EVALUATION")
    base = get_lora_model()
    evaluate_hindi(base, hi_test, tokenizer, "BASE MODEL")
    del base; gc.collect(); torch.cuda.empty_cache()

    model = get_lora_model()
    model = train_sft(model, sft_ds, tokenizer, hi_test)

    ref = get_lora_model()
    ref.load_state_dict(model.state_dict())
    ref.eval()

    model = train_dpo(model, ref, dpo_ds, tokenizer, hi_test)

    logger.info("Training complete.")


if __name__ == "__main__":
    main()
