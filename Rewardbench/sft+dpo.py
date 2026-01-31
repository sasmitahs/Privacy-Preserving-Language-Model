#!/usr/bin/env python3
"""
SFT → DPO with DP-SGD vs PDP-SGD on local LLaMA-3.2-1B-Instruct
Uses YOUR exact hyper-parameters + local paths
Prints the exact table you want
"""
import argparse
import os
import random
import warnings
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Sampler
from torch.optim import AdamW

from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm
from sklearn.utils.extmath import randomized_svd
import logging

# ============================= SETUP =============================
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        handlers=[logging.FileHandler("sft_dpo_comparison.log", mode='w'), logging.StreamHandler()]
    )
    warnings.filterwarnings("ignore")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class FixedBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, drop_last=True, seed=42):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.gen = torch.Generator().manual_seed(seed)
    def __iter__(self):
        n = len(self.dataset)
        idx = torch.randperm(n, generator=self.gen).tolist()
        batches = [idx[i:i + self.batch_size] for i in range(0, n, self.batch_size)]
        return iter([b for b in batches if len(b) == self.batch_size or not self.drop_last])
    def __len__(self):
        return len(self.dataset) // self.batch_size

# ============================= HELPERS =============================
def trainable_params(m): return [p for p in m.parameters() if p.requires_grad]
def flatten_grads(params):
    grads = []
    for p in params:
        if p.grad is not None:
            grads.append(p.grad.view(-1))
        else:
            grads.append(torch.zeros(p.numel(), device=p.device))
    return torch.cat(grads) if grads else torch.tensor([], device="cpu")

# ============================= EVALUATION (RewardBench letter prediction) =============================
@torch.no_grad()
def evaluate(model, loader, tokenizer, device):
    model.eval()
    correct = total = 0
    letters = "ABCD"
    token_ids = torch.tensor([tokenizer.encode(f" {l}", add_special_tokens=False)[0] for l in letters], device=device)

    for batch in loader:
        chosen_ids = batch[0].to(device)
        chosen_mask = batch[1].to(device)
        correct_idx = batch[6].to(device)

        for i in range(chosen_ids.shape[0]):
            labels = batch[2][i]
            nz = (labels != -100).nonzero(as_tuple=True)[0]
            if len(nz) == 0: continue
            prompt_len = nz[0].item()
            ids = chosen_ids[i:i+1, :prompt_len]
            mask = chosen_mask[i:i+1, :prompt_len]
            logits = model(input_ids=ids, attention_mask=mask).logits[:, -1, :]
            pred = logits[0, token_ids].argmax().item()
            true = correct_idx[i].item()
            correct += (pred == true)
            total += 1
    return correct / total if total > 0 else 0.0

# ============================= DPO LOSS =============================
class DPOLoss(nn.Module):
    def __init__(self, beta=0.1): super().__init__(); self.beta = beta
    def forward(self, pc, pr, rc, rr):
        return -F.logsigmoid(self.beta * ((pc - pr) - (rc - rr))).mean()

def get_logps(logits, labels):
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    per_token = -nn.CrossEntropyLoss(reduction='none', ignore_index=-100)(
        shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    per_token = per_token.view(labels.size(0), -1)
    mask = (shift_labels != -100).float()
    return (per_token * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)

# ============================= DATASET (SFT + DPO) =============================
def build_examples(ds, is_dpo=False):
    examples = []
    letters = "ABCD"
    for item in ds:
        chosen = item["chosen"][0] if isinstance(item["chosen"], list) else item["chosen"]
        chosen = chosen.strip()
        rejected = [r.strip() for r in item["rejected"][:3] if r.strip()]
        options = [chosen] + rejected
        while len(options) < 4: options.append("None of the above.")
        random.shuffle(options)
        correct_idx = options.index(chosen)
        correct_letter = letters[correct_idx]
        prompt = f"Question: {item['prompt']}\nOptions:\n" + "\n".join(f"{l}. {o}" for l, o in zip(letters, options)) + "\nAnswer:"

        if is_dpo:
            wrong = random.choice([l for l in letters if l != correct_letter])
            examples.append({"prompt": prompt, "chosen": correct_letter, "rejected": wrong, "correct_idx": correct_idx})
        else:
            examples.append({"prompt": prompt, "response": correct_letter, "correct_idx": correct_idx})
    return examples

def tokenize_sft(examples, tokenizer, max_len):
    ids_l, mask_l, labels_l = [], [], []
    eos = tokenizer.eos_token
    for ex in examples:
        text = ex["prompt"] + " " + ex["response"] + eos
        enc = tokenizer(text, truncation=True, max_length=max_len, padding="max_length", return_tensors="pt")
        ids = enc["input_ids"].squeeze(0)
        mask = enc["attention_mask"].squeeze(0)
        labels = ids.clone()
        prompt_len = len(tokenizer(ex["prompt"], add_special_tokens=False)["input_ids"]) + 1
        labels[:prompt_len] = -100
        ids_l.append(ids); mask_l.append(mask); labels_l.append(labels)
    return TensorDataset(torch.stack(ids_l), torch.stack(mask_l), torch.stack(labels_l))

def tokenize_dpo(examples, tokenizer, max_len):
    c_ids, c_mask, c_lab = [], [], []
    r_ids, r_mask, r_lab = [], [], []
    correct = []
    eos = tokenizer.eos_token
    for ex in examples:
        tc = ex["prompt"] + " " + ex["chosen"] + eos
        enc_c = tokenizer(tc, truncation=True, max_length=max_len, padding="max_length", return_tensors="pt")
        ci = enc_c["input_ids"].squeeze(0); cm = enc_c["attention_mask"].squeeze(0)
        cl = ci.clone()
        pl = len(tokenizer(ex["prompt"], add_special_tokens=False)["input_ids"]) + 1
        cl[:pl] = -100

        tr = ex["prompt"] + " " + ex["rejected"] + eos
        enc_r = tokenizer(tr, truncation=True, max_length=max_len, padding="max_length", return_tensors="pt")
        ri = enc_r["input_ids"].squeeze(0); rm = enc_r["attention_mask"].squeeze(0)
        rl = ri.clone(); rl[:pl] = -100

        c_ids.append(ci); c_mask.append(cm); c_lab.append(cl)
        r_ids.append(ri); r_mask.append(rm); r_lab.append(rl)
        correct.append(torch.tensor(ex["correct_idx"]))

    return TensorDataset(
        torch.stack(c_ids), torch.stack(c_mask), torch.stack(c_lab),
        torch.stack(r_ids), torch.stack(r_mask), torch.stack(r_lab),
        torch.stack(correct),
        torch.zeros(len(examples), dtype=torch.long)
    )

# ============================= GRADIENT & GSD =============================
def compute_grad_matrix_sft(model, loader, device, C):
    grads = []
    params = trainable_params(model)
    for batch in loader:
        for i in range(batch[0].size(0)):
            model.zero_grad()
            loss = model(input_ids=batch[0][i:i+1].to(device),
                         attention_mask=batch[1][i:i+1].to(device),
                         labels=batch[2][i:i+1].to(device)).loss
            loss.backward()
            g = flatten_grads(params)
            g = g * min(1.0, C / (g.norm() + 1e-12))
            grads.append(g.cpu())
            model.zero_grad()
    return np.stack(grads) if grads else np.zeros((0, sum(p.numel() for p in params)))

def compute_grad_matrix_dpo(model, ref, loader, device, C):
    grads = []
    params = trainable_params(model)
    loss_fn = DPOLoss()
    for batch in loader:
        for i in range(batch[0].size(0)):
            model.zero_grad()
            pc = get_logps(model(input_ids=batch[0][i:i+1].to(device), attention_mask=batch[1][i:i+1].to(device)).logits, batch[2][i:i+1].to(device))
            pr = get_logps(model(input_ids=batch[3][i:i+1].to(device), attention_mask=batch[4][i:i+1].to(device)).logits, batch[5][i:i+1].to(device))
            with torch.no_grad():
                rc = get_logps(ref(input_ids=batch[0][i:i+1].to(device), attention_mask=batch[1][i:i+1].to(device)).logits, batch[2][i:i+1].to(device))
                rr = get_logps(ref(input_ids=batch[3][i:i+1].to(device), attention_mask=batch[4][i:i+1].to(device)).logits, batch[5][i:i+1].to(device))
            loss = loss_fn(pc, pr, rc, rr)
            loss.backward()
            g = flatten_grads(params)
            g = g * min(1.0, C / (g.norm() + 1e-12))
            grads.append(g.cpu())
            model.zero_grad()
    return np.stack(grads) if grads else np.zeros((0, sum(p.numel() for p in params)))

def gsd(G1, G2, k=20):
    if G1.shape[0] == 0 or G2.shape[0] == 0: return float('inf')
    _, _, V1 = randomized_svd(G1, n_components=k, random_state=42)
    _, _, V2 = randomized_svd(G2, n_components=k, random_state=42)
    return np.sqrt(k - np.sum(np.linalg.svd(V1 @ V2.T, compute_uv=False)**2))

# ============================= TRAINING (your exact hyper-params) =============================
def train_dp_sgd_sft(model, loader, test_loader, tokenizer, device, C, lr, epochs, sigma):
    opt = AdamW(trainable_params(model), lr=lr)
    params = trainable_params(model)
    sizes = [p.numel() for p in params]
    shapes = [p.shape for p in params]
    for epoch in range(epochs):
        model.train()
        for batch in tqdm(loader, desc=f"SFT DP-SGD {epoch+1}/{epochs}"):
            per_sample = []
            for i in range(batch[0].size(0)):
                model.zero_grad()
                loss = model(input_ids=batch[0][i:i+1].to(device), attention_mask=batch[1][i:i+1].to(device),
                             labels=batch[2][i:i+1].to(device)).loss
                loss.backward()
                g = flatten_grads(params) * min(1.0, C / (flatten_grads(params).norm() + 1e-12))
                per_sample.append(g)
                model.zero_grad()
            if not per_sample: continue
            g_avg = torch.stack(per_sample).mean(0)
            noise = torch.normal(0, sigma * C, g_avg.shape, device=device)
            g_noisy = g_avg + noise
            idx = 0
            for p, sz, sh in zip(params, sizes, shapes):
                p.grad = g_noisy[idx:idx+sz].view(sh)
                idx += sz
            opt.step(); opt.zero_grad()
    return evaluate(model, test_loader, tokenizer, device)

def train_pdp_sgd_sft(model, loader, test_loader, tokenizer, device, C, lr, total_steps, sigma, V_k):
    opt = AdamW(trainable_params(model), lr=lr)
    params = trainable_params(model)
    sizes = [p.numel() for p in params]
    step = 0
    it = iter(loader)
    pbar = tqdm(total=total_steps, desc="PDP-SFT")
    while step < total_steps:
        step += 1; pbar.update(1)
        try: batch = next(it)
        except StopIteration: it = iter(loader); batch = next(it)
        per_sample = []
        for i in range(batch[0].size(0)):
            model.zero_grad()
            loss = model(input_ids=batch[0][i:i+1].to(device), attention_mask=batch[1][i:i+1].to(device),
                         labels=batch[2][i:i+1].to(device)).loss
            loss.backward()
            g = flatten_grads(params) * min(1.0, C / (flatten_grads(params).norm() + 1e-12))
            per_sample.append(g)
            model.zero_grad()
        if not per_sample: continue
        g_avg = torch.stack(per_sample).mean(0)
        noise = torch.normal(0, sigma * C, g_avg.shape, device=device)
        g_noisy = g_avg + noise
        g_proj = V_k @ (V_k.T @ g_noisy)
        idx = 0
        for p, sz in zip(params, sizes):
            p.grad = g_proj[idx:idx+sz].view(p.shape)
            idx += sz
        opt.step(); opt.zero_grad()
    pbar.close()
    return evaluate(model, test_loader, tokenizer, device)

def train_dp_sgd_dpo(model, ref, loader, test_loader, tokenizer, device, C, lr, epochs, sigma):
    opt = AdamW(trainable_params(model), lr=lr)
    loss_fn = DPOLoss()
    params = trainable_params(model)
    sizes = [p.numel() for p in params]
    shapes = [p.shape for p in params]
    for epoch in range(epochs):
        model.train(); ref.eval()
        for batch in tqdm(loader, desc=f"DPO DP-SGD {epoch+1}/{epochs}"):
            per_sample = []
            for i in range(batch[0].size(0)):
                model.zero_grad()
                pc = get_logps(model(input_ids=batch[0][i:i+1].to(device), attention_mask=batch[1][i:i+1].to(device)).logits, batch[2][i:i+1].to(device))
                pr = get_logps(model(input_ids=batch[3][i:i+1].to(device), attention_mask=batch[4][i:i+1].to(device)).logits, batch[5][i:i+1].to(device))
                with torch.no_grad():
                    rc = get_logps(ref(input_ids=batch[0][i:i+1].to(device), attention_mask=batch[1][i:i+1].to(device)).logits, batch[2][i:i+1].to(device))
                    rr = get_logps(ref(input_ids=batch[3][i:i+1].to(device), attention_mask=batch[4][i:i+1].to(device)).logits, batch[5][i:i+1].to(device))
                loss = loss_fn(pc, pr, rc, rr)
                loss.backward()
                g = flatten_grads(params) * min(1.0, C / (flatten_grads(params).norm() + 1e-12))
                per_sample.append(g)
                model.zero_grad()
            if not per_sample: continue
            g_avg = torch.stack(per_sample).mean(0)
            noise = torch.normal(0, sigma * C, g_avg.shape, device=device)
            g_noisy = g_avg + noise
            idx = 0
            for p, sz, sh in zip(params, sizes, shapes):
                p.grad = g_noisy[idx:idx+sz].view(sh)
                idx += sz
            opt.step(); opt.zero_grad()
    return evaluate(model, test_loader, tokenizer, device)

def train_pdp_sgd_dpo(model, ref, loader, test_loader, tokenizer, device, C, lr, total_steps, sigma, V_k):
    opt = AdamW(trainable_params(model), lr=lr)
    loss_fn = DPOLoss()
    params = trainable_params(model)
    sizes = [p.numel() for p in params]
    step = 0
    it = iter(loader)
    pbar = tqdm(total=total_steps, desc="PDP-DPO")
    while step < total_steps:
        step += 1; pbar.update(1)
        try: batch = next(it)
        except StopIteration: it = iter(loader); batch = next(it)
        per_sample = []
        for i in range(batch[0].size(0)):
            model.zero_grad()
            pc = get_logps(model(input_ids=batch[0][i:i+1].to(device), attention_mask=batch[1][i:i+1].to(device)).logits, batch[2][i:i+1].to(device))
            pr = get_logps(model(input_ids=batch[3][i:i+1].to(device), attention_mask=batch[4][i:i+1].to(device)).logits, batch[5][i:i+1].to(device))
            with torch.no_grad():
                rc = get_logps(ref(input_ids=batch[0][i:i+1].to(device), attention_mask=batch[1][i:i+1].to(device)).logits, batch[2][i:i+1].to(device))
                rr = get_logps(ref(input_ids=batch[3][i:i+1].to(device), attention_mask=batch[4][i:i+1].to(device)).logits, batch[5][i:i+1].to(device))
            loss = loss_fn(pc, pr, rc, rr)
            loss.backward()
            g = flatten_grads(params) * min(1.0, C / (flatten_grads(params).norm() + 1e-12))
            per_sample.append(g)
            model.zero_grad()
        if not per_sample: continue
        g_avg = torch.stack(per_sample).mean(0)
        noise = torch.normal(0, sigma * C, g_avg.shape, device=device)
        g_noisy = g_avg + noise
        g_proj = V_k @ (V_k.T @ g_noisy)
        idx = 0
        for p, sz in zip(params, sizes):
            p.grad = g_proj[idx:idx+sz].view(p.shape)
            idx += sz
        opt.step(); opt.zero_grad()
    pbar.close()
    return evaluate(model, test_loader, tokenizer, device)

# ============================= MODEL & CLONE =============================
def build_model(model_dir):
    base = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16, local_files_only=True, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    lora_cfg = LoraConfig(r=64, lora_alpha=16, target_modules=["q_proj","k_proj","v_proj","o_proj"],
                          lora_dropout=0.05, bias="none", task_type=TaskType.CAUSAL_LM)
    return get_peft_model(base, lora_cfg), tokenizer

def clone_peft_model(src):
    new_base = AutoModelForCausalLM.from_config(src.base_model.config)
    new_peft = get_peft_model(new_base, src.peft_config[list(src.peft_config.keys())[0]])
    new_peft.load_state_dict(src.state_dict(), strict=False)
    return new_peft.to(src.device)

# ============================= MAIN (your exact args) =============================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="./models/llama-3.2-1b-instruct")
    parser.add_argument("--dataset_dir", default="./datasets/reward_bench_dataset")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--C", type=float, default=5.0)
    parser.add_argument("--sigma", type=float, default=10.0)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--max_length", type=int, default=3048)
    args = parser.parse_args()

    setup_logging()
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Device: {device}")

    ds = load_from_disk(args.dataset_dir).shuffle(seed=42)
    N = len(ds)
    test_N = N // 16
    private_N = 10 * test_N
    public_N = 5 * test_N

    private = ds.select(range(private_N))
    public = ds.select(range(private_N, private_N + public_N))
    test = ds.select(range(private_N + public_N, private_N + public_N + test_N))

    base_peft, tokenizer = build_model(args.model_dir)
    base_peft = base_peft.to(device)

    private_sft_ex = build_examples(private, is_dpo=False)
    private_dpo_ex = build_examples(private, is_dpo=True)
    test_ex = build_examples(test, is_dpo=True)

    private_sft_ds = tokenize_sft(private_sft_ex, tokenizer, args.max_length)
    private_dpo_ds = tokenize_dpo(private_dpo_ex, tokenizer, args.max_length)
    test_ds = tokenize_dpo(test_ex, tokenizer, args.max_length)

    private_sft_loader = DataLoader(private_sft_ds, batch_sampler=FixedBatchSampler(private_sft_ds, args.batch_size))
    private_dpo_loader = DataLoader(private_dpo_ds, batch_sampler=FixedBatchSampler(private_dpo_ds, args.batch_size))
    test_loader = DataLoader(test_ds, batch_sampler=FixedBatchSampler(test_ds, args.batch_size, drop_last=False))

    total_steps = args.epochs * len(private_sft_loader)

    # DP-SGD baseline
    model_sft_dp = clone_peft_model(base_peft)
    sft_dp_acc = train_dp_sgd_sft(model_sft_dp, private_sft_loader, test_loader, tokenizer, device,
                                  args.C, args.lr, args.epochs, args.sigma)

    ref_dp = clone_peft_model(model_sft_dp); ref_dp.eval()
    model_dpo_dp = clone_peft_model(model_sft_dp)
    dpo_dp_acc = train_dp_sgd_dpo(model_dpo_dp, ref_dp, private_dpo_loader, test_loader, tokenizer, device,
                                  args.C, args.lr, args.epochs, args.sigma)

    G_priv_sft = compute_grad_matrix_sft(base_peft, private_sft_loader, device, args.C)
    G_priv_dpo = compute_grad_matrix_dpo(model_sft_dp, ref_dp, private_dpo_loader, device, args.C)

    results = []
    subset_size = len(public) // 5
    for i in range(5):
        start = i * subset_size
        end = (i + 1) * subset_size if i < 4 else len(public)
        subset = public.select(range(start, end))

        sft_ex = build_examples(subset, False)
        dpo_ex = build_examples(subset, True)
        sft_ds = tokenize_sft(sft_ex, tokenizer, args.max_length)
        dpo_ds = tokenize_dpo(dpo_ex, tokenizer, args.max_length)
        sft_loader = DataLoader(sft_ds, batch_sampler=FixedBatchSampler(sft_ds, args.batch_size))
        dpo_loader = DataLoader(dpo_ds, batch_sampler=FixedBatchSampler(dpo_ds, args.batch_size))

        G_pub_sft = compute_grad_matrix_sft(base_peft, sft_loader, device, args.C)
        G_pub_dpo = compute_grad_matrix_dpo(model_sft_dp, ref_dp, dpo_loader, device, args.C)
        sft_gsd = gsd(G_priv_sft, G_pub_sft, args.k)
        dpo_gsd = gsd(G_priv_dpo, G_pub_dpo, args.k)

        # PDP-SFT
        _, _, Vt = randomized_svd(G_pub_sft, n_components=args.k, random_state=42)
        V_k_sft = torch.from_numpy(Vt.T).float().to(device)
        model_sft_pdp = clone_peft_model(base_peft)
        sft_pdp_acc = train_pdp_sgd_sft(model_sft_pdp, private_sft_loader, test_loader, tokenizer, device,
                                        args.C, args.lr, total_steps, args.sigma, V_k_sft)

        # PDP-DPO
        ref_pdp = clone_peft_model(model_sft_pdp); ref_pdp.eval()
        G_pub_dpo_pdp = compute_grad_matrix_dpo(model_sft_pdp, ref_pdp, dpo_loader, device, args.C)
        _, _, Vt_dpo = randomized_svd(G_pub_dpo_pdp, n_components=args.k, random_state=42)
        V_k_dpo = torch.from_numpy(Vt_dpo.T).float().to(device)
        model_dpo_pdp = clone_peft_model(model_sft_pdp)
        dpo_pdp_acc = train_pdp_sgd_dpo(model_dpo_pdp, ref_pdp, private_dpo_loader, test_loader, tokenizer, device,
                                        args.C, args.lr, total_steps, args.sigma, V_k_dpo)

        results.append({
            "name": f"subset_{i}",
            "sft_gsd": sft_gsd,
            "dpo_gsd": dpo_gsd,
            "sft_pdp": sft_pdp_acc,
            "sft_dp": sft_dp_acc,
            "dpo_pdp": dpo_pdp_acc,
            "dpo_dp": dpo_dp_acc
        })

    print("\n| Subset   | SFT GSD | DPO GSD | SFT PDP | SFT DP | DPO PDP | DPO DP |")
    print("|----------|---------|---------|---------|--------|---------|--------|")
    for r in results:
        print(f"| {r['name']:<8} | {r['sft_gsd']:7.3f} | {r['dpo_gsd']:7.3f} | {r['sft_pdp']:7.3f} | {r['sft_dp']:6.3f} | {r['dpo_pdp']:7.3f} | {r['dpo_dp']:6.3f} |")
    print("="*80)

if __name__ == "__main__":
    main()
