#!/usr/bin/env python3
"""
RewardBench2 - FULLY FIXED PDP-SGD DPO for LLaMA-3.2-1B-Instruct

"""
import argparse
import json
import math
import os
import random
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
from typing import List, Dict, Any
from torch.utils.data import DataLoader, Dataset, TensorDataset, Sampler
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_from_disk
from tqdm import tqdm
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.utils.extmath import randomized_svd
import logging
from collections import Counter

# -----------------------
# Logging & Setup
# -----------------------
def setup_logging(log_file="training_fixed_pdp_dpo_llama.log"):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode='w'),
            logging.StreamHandler()
        ]
    )
    warnings.filterwarnings("ignore")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class FixedBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, drop_last=True, seed=42):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.generator = torch.Generator().manual_seed(seed)

    def __iter__(self):
        if not self.dataset:
            return iter([])
        n = len(self.dataset)
        idxs = torch.randperm(n, generator=self.generator).tolist()
        batches = [idxs[i:i+self.batch_size] for i in range(0, n, self.batch_size)]
        return iter([b for b in batches if len(b) == self.batch_size or not self.drop_last])

    def __len__(self):
        if not self.dataset:
            return 0
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

# -----------------------
# Validation
# -----------------------
def valid_example(ex):
    try:
        return (
            "prompt" in ex and isinstance(ex["prompt"], str) and ex["prompt"].strip() and
            "chosen" in ex and (
                (isinstance(ex["chosen"], str) and ex["chosen"].strip()) or
                (isinstance(ex["chosen"], list) and len(ex["chosen"]) > 0 and ex["chosen"][0].strip())
            ) and
            "rejected" in ex and isinstance(ex["rejected"], list) and len(ex["rejected"]) >= 1 and
            any(isinstance(opt, str) and opt.strip() for opt in ex["rejected"]) and
            "subset" in ex and isinstance(ex["subset"], str) and ex["subset"].strip()
        )
    except:
        return False

# -----------------------
# Evaluation WITH DEBUG
# -----------------------
@torch.no_grad()
def evaluate_rewardbench_accuracy(model, loader, tokenizer, device):
    model.eval()
    correct_by_domain = {"chat": 0, "chat-hard": 0, "safety": 0, "reasoning": 0, "instruction-following": 0, "ties": 0, "factuality": 0, "unknown": 0}
    total_by_domain = {k: 0 for k in correct_by_domain}
    ties_scores = []
    letters = ['A', 'B', 'C', 'D']

    # FIXED: leading space for LLaMA tokenizer
    completions = [f" {l}" for l in letters]
    token_ids = []
    for c in completions:
        toks = tokenizer.encode(c, add_special_tokens=False)
        tid = toks[0] if toks else tokenizer.pad_token_id
        token_ids.append(tid)
    token_ids = torch.tensor(token_ids, device=device)

    

    def get_prob(input_ids, attention_mask, token_id):
        if token_id == tokenizer.pad_token_id:
            return 0.0
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[:, -1, :]
        return F.softmax(logits, dim=-1)[0, token_id].item()

    total_evaluated = 0
    for batch in loader:
        chosen_input_ids = batch[0].to(device)
        chosen_attention_mask = batch[1].to(device)
        chosen_labels = batch[2].to(device)
        correct_idx = batch[6].to(device)
        domains = batch[7]

        for i in range(chosen_input_ids.size(0)):
            labels_i = chosen_labels[i]
            non_ignore = (labels_i != -100).nonzero(as_tuple=True)[0]
            if len(non_ignore) == 0: continue
            prompt_len = non_ignore[0].item()

            input_ids = chosen_input_ids[i][:prompt_len].unsqueeze(0)
            attention_mask = chosen_attention_mask[i][:prompt_len].unsqueeze(0)
            domain_idx = domains[i].item()
            domain = list(correct_by_domain.keys())[domain_idx]

            probs = [get_prob(input_ids, attention_mask, tid) for tid in token_ids]
            pred_idx = int(np.argmax(probs))
            true_idx = int(correct_idx[i].item())

            total_by_domain[domain] += 1
            total_evaluated += 1

            if total_evaluated % 50 == 0:
                prompt_snippet = tokenizer.decode(chosen_input_ids[i][:prompt_len], skip_special_tokens=True)[-120:]
                
            if domain == "ties":
                chosen_probs = [probs[true_idx]]
                rejected_probs = [probs[j] for j in range(4) if j != true_idx]
                max_c, max_r = max(chosen_probs), max(rejected_probs)
                margin = min((max_c - max_r) / 0.5, 1.0) if max_c > max_r else 0.0
                ties_scores.append(1.0 + margin)
            else:
                if pred_idx == true_idx:
                    correct_by_domain[domain] += 1

    domain_accuracies = {}
    for domain in correct_by_domain:
        total = total_by_domain[domain]
        if total > 0:
            if domain == "ties":
                domain_accuracies[domain] = sum(ties_scores) / len(ties_scores) / 2.0 if ties_scores else 0.0
            else:
                domain_accuracies[domain] = correct_by_domain[domain] / total
        else:
            domain_accuracies[domain] = 0.0

    valid_domains = [d for d in total_by_domain if total_by_domain[d] > 0]
    overall = sum(domain_accuracies[d] for d in valid_domains) / len(valid_domains) if valid_domains else 0.0

    logging.info(f"FINAL DOMAIN ACCURACIES: {domain_accuracies}")
    logging.info(f"OVERALL ACCURACY: {overall:.4f}")
    return {"domain_accuracies": domain_accuracies, "overall_accuracy": overall}

# -----------------------
# DPO Loss
# -----------------------
class DPOLoss(nn.Module):
    def __init__(self, beta=0.1):
        super().__init__()
        self.beta = beta
    def forward(self, policy_chosen_logps, policy_rejected_logps, ref_chosen_logps, ref_rejected_logps):
        logits = self.beta * ((policy_chosen_logps - policy_rejected_logps) - (ref_chosen_logps - ref_rejected_logps))
        return -F.logsigmoid(logits).mean()

def get_batch_logps(logits, labels):
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss_fct = nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
    per_token_logps = -loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    per_token_logps = per_token_logps.view(labels.size(0), -1)
    mask = (shift_labels != -100).float()
    return (per_token_logps * mask).sum(-1) / mask.sum(-1).clamp(min=1)

# -----------------------
# Dataset Helpers - FIXED WITH LEADING SPACE
# -----------------------
def format_rewardbench_prompt(prompt: str, options: List[str]) -> str:
    letters = ['A', 'B', 'C', 'D']
    opt_text = "\n".join([f"{letter}. {opt}" for letter, opt in zip(letters, options)])
    return f"This is a multiple-choice question. Select the best completion by providing only the single capital letter (A, B, C, or D) corresponding to the correct answer. Prompt: {prompt}\nOptions:\n{opt_text}\nAnswer: "

def build_dpo_examples_from_hf(ds, num_examples: int = None):
    domains = set(ds["subset"]) if "subset" in ds.features else set()
    examples_by_domain = {d.lower(): [] for d in domains}
    examples_by_domain["unknown"] = []
    skipped = Counter()
    letters = ['A', 'B', 'C', 'D']

    for i, item in enumerate(ds):
        if num_examples is not None and i >= num_examples: break
        if not valid_example(item):
            skipped['invalid'] += 1
            continue

        prompt = item.get("prompt", "")
        chosen = item["chosen"]
        if isinstance(chosen, list):
            chosen = chosen[0].strip() if chosen else ""
        rejected = item.get("rejected", [])
        options = [chosen] + [r.strip() for r in rejected if r.strip()][:3]
        while len(options) < 4:
            options.append("Placeholder")
            skipped['padded'] += 1
        random.shuffle(options)
        try:
            correct_idx = options.index(chosen)
        except ValueError:
            skipped['not_found'] += 1
            continue

        ans = letters[correct_idx]
        rejected_letter = random.choice([l for l in letters if l != ans])
        domain = item.get("subset", "unknown").lower()
        if domain not in examples_by_domain:
            domain = "unknown"

        prompt_text = format_rewardbench_prompt(prompt, options)
        examples_by_domain[domain].append({
            "prompt": prompt_text,
            "chosen": ans,
            "rejected": rejected_letter,
            "correct_idx": correct_idx,
            "domain": domain
        })

    logging.info(f"Built DPO examples | Total: {sum(len(v) for v in examples_by_domain.values())} | Skipped: {dict(skipped)}")
    return examples_by_domain

def build_dpo_dataset(examples_by_domain, tokenizer, max_length=5160):
    enc = []
    domain_indices = []
    domain_map = {d: i for i, d in enumerate(sorted(examples_by_domain.keys()))}
    eos = tokenizer.eos_token or "<|eot_id|>"

    for domain, exs in examples_by_domain.items():
        for ex in exs:
            prompt_text = ex["prompt"]
            # CRITICAL FIX: Add leading space so model learns to output " A", " B", etc.
            chosen_cont = " " + ex["chosen"] + eos
            rejected_cont = " " + ex["rejected"] + eos

            enc_c = tokenizer(prompt_text + chosen_cont, truncation=True, padding="max_length", max_length=max_length, return_tensors="pt")
            enc_r = tokenizer(prompt_text + rejected_cont, truncation=True, padding="max_length", max_length=max_length, return_tensors="pt")

            chosen_ids = enc_c["input_ids"].squeeze(0)
            chosen_mask = enc_c["attention_mask"].squeeze(0)
            chosen_labels = chosen_ids.clone()
            prompt_len = len(tokenizer.encode(prompt_text, add_special_tokens=False))
            chosen_labels[:prompt_len] = -100
            chosen_labels[chosen_ids == tokenizer.pad_token_id] = -100

            rej_ids = enc_r["input_ids"].squeeze(0)
            rej_mask = enc_r["attention_mask"].squeeze(0)
            rej_labels = rej_ids.clone()
            rej_labels[:prompt_len] = -100
            rej_labels[rej_ids == tokenizer.pad_token_id] = -100

            enc.append({
                "chosen_input_ids": chosen_ids,
                "chosen_attention_mask": chosen_mask,
                "chosen_labels": chosen_labels,
                "rejected_input_ids": rej_ids,
                "rejected_attention_mask": rej_mask,
                "rejected_labels": rej_labels,
                "correct_idx": torch.tensor(ex["correct_idx"], dtype=torch.long),
                "domain": domain_map[domain]
            })
            domain_indices.append(domain_map[domain])

    dataset = TensorDataset(
        torch.stack([e["chosen_input_ids"] for e in enc]),
        torch.stack([e["chosen_attention_mask"] for e in enc]),
        torch.stack([e["chosen_labels"] for e in enc]),
        torch.stack([e["rejected_input_ids"] for e in enc]),
        torch.stack([e["rejected_attention_mask"] for e in enc]),
        torch.stack([e["rejected_labels"] for e in enc]),
        torch.stack([e["correct_idx"] for e in enc]),
        torch.tensor(domain_indices, dtype=torch.long)
    )
    logging.info(f"DPO Dataset built: {len(enc)} examples")
    return dataset

# -----------------------
# Training Functions
# -----------------------
def train_pdp_sgd_dpo_fixed(model, ref_model, private_loader, test_loader, tokenizer, device,
                            max_grad_norm, lr, V_k, total_steps=1000, sigma=10.0):
    optimizer = AdamW(trainable_params(model), lr=lr)
    dpo_loss_fn = DPOLoss(beta=0.1)
    tparams = trainable_params(model)
    param_sizes = [p.numel() for p in tparams]

    private_iter = iter(private_loader)
    step = 0
    steps_per_epoch = len(private_loader)  # <-- this is the number of batches per epoch
    pbar = tqdm(total=total_steps, desc="PDP-SGD DPO (fixed V_k)")

    epoch = 0
    while step < total_steps:
        epoch_start_step = step

        # Run one full epoch
        while step < total_steps and (step - epoch_start_step) < steps_per_epoch:
            step += 1
            pbar.update(1)

            try:
                batch = next(private_iter)
            except StopIteration:
                private_iter = iter(private_loader)
                batch = next(private_iter)

            per_sample_grads = []
            for i in range(batch[0].size(0)):
                model.zero_grad(set_to_none=True)
                c_out = model(input_ids=batch[0][i:i+1].to(device),
                              attention_mask=batch[1][i:i+1].to(device))
                r_out = model(input_ids=batch[3][i:i+1].to(device),
                              attention_mask=batch[4][i:i+1].to(device))
                pc = get_batch_logps(c_out.logits, batch[2][i:i+1].to(device))
                pr = get_batch_logps(r_out.logits, batch[5][i:i+1].to(device))
                with torch.no_grad():
                    rc = get_batch_logps(ref_model(input_ids=batch[0][i:i+1].to(device),
                                            attention_mask=batch[1][i:i+1].to(device)).logits,
                                         batch[2][i:i+1].to(device))
                    rr = get_batch_logps(ref_model(input_ids=batch[3][i:i+1].to(device),
                                            attention_mask=batch[4][i:i+1].to(device)).logits,
                                         batch[5][i:i+1].to(device))
                loss = dpo_loss_fn(pc, pr, rc, rr)
                if loss > 100:
                    loss = loss.clamp(max=100)
                loss.backward()
                g = flatten_grads(tparams)
                g_norm = torch.norm(g).item()
                if g_norm > 0:
                    g = g * min(1.0, max_grad_norm / g_norm)
                per_sample_grads.append(g)
                model.zero_grad(set_to_none=True)

            if not per_sample_grads:
                continue

            g_t = torch.stack(per_sample_grads).mean(0)
            noise = torch.normal(0, sigma * max_grad_norm, size=g_t.shape, device=device)
            g_noisy = g_t + noise
            g_tilde = V_k @ (V_k.T @ g_noisy)

            idx = 0
            for p, sz in zip(tparams, param_sizes):
                p.grad = g_tilde[idx:idx + sz].view(p.shape)
                idx += sz
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        # ---------- END OF EPOCH → REPORT ACCURACY ----------
        epoch += 1
        acc = evaluate_rewardbench_accuracy(model, test_loader, tokenizer, device)
        logging.info(f"PDP-SGD DPO Epoch {epoch} | Accuracy: {acc['overall_accuracy']:.4f}")

    pbar.close()
    final_acc = evaluate_rewardbench_accuracy(model, test_loader, tokenizer, device)
    logging.info(f"PDP-SGD DPO Training finished | Final Accuracy: {final_acc['overall_accuracy']:.4f}")
    return final_acc


def train_dp_sgd_dpo(model, ref_model, loader, test_loader, tokenizer, device, max_grad_norm, lr, epochs=3, sigma=10.0):
    optimizer = AdamW(trainable_params(model), lr=lr)
    dpo_loss_fn = DPOLoss(beta=0.1)
    tparams = trainable_params(model)
    param_sizes = [p.numel() for p in tparams]
    param_shapes = [p.shape for p in tparams]

    for epoch in range(epochs):
        model.train()
        for batch in tqdm(loader, desc=f"DP-SGD DPO Epoch {epoch+1}/{epochs}"):
            per_sample_grads = []
            for i in range(batch[0].size(0)):
                model.zero_grad(set_to_none=True)
                c_out = model(input_ids=batch[0][i:i+1].to(device), attention_mask=batch[1][i:i+1].to(device))
                r_out = model(input_ids=batch[3][i:i+1].to(device), attention_mask=batch[4][i:i+1].to(device))
                pc = get_batch_logps(c_out.logits, batch[2][i:i+1].to(device))
                pr = get_batch_logps(r_out.logits, batch[5][i:i+1].to(device))
                with torch.no_grad():
                    rc = get_batch_logps(ref_model(input_ids=batch[0][i:i+1].to(device), attention_mask=batch[1][i:i+1].to(device)).logits, batch[2][i:i+1].to(device))
                    rr = get_batch_logps(ref_model(input_ids=batch[3][i:i+1].to(device), attention_mask=batch[4][i:i+1].to(device)).logits, batch[5][i:i+1].to(device))
                loss = dpo_loss_fn(pc, pr, rc, rr)
                if loss > 100: loss = loss.clamp(max=100)
                loss.backward()
                g = flatten_grads(tparams)
                g_norm = torch.norm(g).item()
                if g_norm > 0:
                    g = g * min(1.0, max_grad_norm / g_norm)
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

        acc = evaluate_rewardbench_accuracy(model, test_loader, tokenizer, device)
        logging.info(f"DP-SGD DPO Epoch {epoch+1} finished | Accuracy: {acc['overall_accuracy']:.4f}")

    return acc

def compute_gradient_matrix_dpo(model, ref_model, loader, device, max_grad_norm):
    grads = []
    tparams = trainable_params(model)
    dpo_loss_fn = DPOLoss(beta=0.1)
    for batch in loader:
        for i in range(batch[0].size(0)):
            model.zero_grad(set_to_none=True)
            c_out = model(input_ids=batch[0][i:i+1].to(device), attention_mask=batch[1][i:i+1].to(device))
            r_out = model(input_ids=batch[3][i:i+1].to(device), attention_mask=batch[4][i:i+1].to(device))
            pc = get_batch_logps(c_out.logits, batch[2][i:i+1].to(device))
            pr = get_batch_logps(r_out.logits, batch[5][i:i+1].to(device))
            with torch.no_grad():
                rc = get_batch_logps(ref_model(input_ids=batch[0][i:i+1].to(device), attention_mask=batch[1][i:i+1].to(device)).logits, batch[2][i:i+1].to(device))
                rr = get_batch_logps(ref_model(input_ids=batch[3][i:i+1].to(device), attention_mask=batch[4][i:i+1].to(device)).logits, batch[5][i:i+1].to(device))
            loss = dpo_loss_fn(pc, pr, rc, rr)
            if loss is None or torch.isnan(loss): continue
            loss.backward()
            g = flatten_grads(tparams)
            g_norm = torch.norm(g).item()
            if g_norm > 0:
                g = g * min(1.0, max_grad_norm / g_norm)
            grads.append(g.cpu())
            model.zero_grad(set_to_none=True)
    return torch.stack(grads).numpy()

def gradient_subspace_distance(G1, G2, k=20):
    _, _, V1 = randomized_svd(G1, n_components=k, random_state=42)
    _, _, V2 = randomized_svd(G2, n_components=k, random_state=42)
    M = V1 @ V2.T
    s = np.linalg.svd(M, compute_uv=False)
    return np.sqrt(k - np.sum(s ** 2))

# -----------------------
# Model
# -----------------------
def build_llama_lora(local_path):
    model = AutoModelForCausalLM.from_pretrained(
        local_path,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    if model.config.pad_token_id is None:
        model.config.pad_token_id = model.config.eos_token_id
    cfg = LoraConfig(
        r=8, lora_alpha=8,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.1, bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    return get_peft_model(model, cfg)

def now_ts():
    return datetime.now().strftime("%Y%m%dT%H%M%S")

# -----------------------
# Main
# -----------------------
def main():
    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="./models/llama-3.2-1b-instruct")
    parser.add_argument("--dataset_dir", default="./datasets/reward_bench_dataset")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max_grad_norm", type=float, default=5.0)
    parser.add_argument("--sigma", type=float, default=10.0)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--max_length", type=int, default=3160)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    for k, v in vars(args).items():
        logging.info(f"{k}: {v}")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    ds = load_from_disk(args.dataset_dir).shuffle(seed=args.seed)
    N = len(ds)
    test_N = N // 16
    private_N = 10 * test_N
    public_total_N = 5 * test_N

    private = ds.select(range(private_N))
    public = ds.select(range(private_N, private_N + public_total_N))
    test = ds.select(range(private_N + public_total_N, private_N + public_total_N + test_N))

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    private_ex = build_dpo_examples_from_hf(private)
    test_ex = build_dpo_examples_from_hf(test)
    private_ds = build_dpo_dataset(private_ex, tokenizer, args.max_length)
    test_ds = build_dpo_dataset(test_ex, tokenizer, args.max_length)

    private_loader = DataLoader(private_ds, batch_sampler=FixedBatchSampler(private_ds, args.batch_size))
    test_loader = DataLoader(test_ds, batch_sampler=FixedBatchSampler(test_ds, args.batch_size, drop_last=False))

    init_model = build_llama_lora(args.model_dir).to(device)
    ref_model = build_llama_lora(args.model_dir).to(device)
    ref_model.load_state_dict(init_model.state_dict())
    ref_model.eval()

    # DP-SGD DPO baseline
    model_dp = build_llama_lora(args.model_dir).to(device)
    model_dp.load_state_dict(init_model.state_dict())
    dp_acc = train_dp_sgd_dpo(model_dp, ref_model, private_loader, test_loader, tokenizer, device,
                              args.max_grad_norm, args.lr, epochs=args.epochs, sigma=args.sigma)['overall_accuracy']
    torch.save(model_dp.state_dict(), f"dpo_dp_sigma{args.sigma}.pth")

    # Compute private gradient matrix
    G_priv = compute_gradient_matrix_dpo(init_model, ref_model, private_loader, device, args.max_grad_norm)

    results = []
    subset_size = len(public) // 5
    total_steps = args.epochs * (len(private_ds) // args.batch_size)

    for i in range(5):
        start = i * subset_size
        end = (i + 1) * subset_size if i < 4 else len(public)
        subset = public.select(range(start, end))
        subset_ex = build_dpo_examples_from_hf(subset)
        subset_ds = build_dpo_dataset(subset_ex, tokenizer, args.max_length)
        subset_loader = DataLoader(subset_ds, batch_sampler=FixedBatchSampler(subset_ds, args.batch_size))

        G_pub = compute_gradient_matrix_dpo(init_model, ref_model, subset_loader, device, args.max_grad_norm)
        gsd = gradient_subspace_distance(G_priv, G_pub, args.k)

        _, _, Vt = randomized_svd(G_pub, n_components=args.k, random_state=42)
        V_k = torch.from_numpy(Vt.T).float().to(device)

        model_pdp = build_llama_lora(args.model_dir).to(device)
        model_pdp.load_state_dict(init_model.state_dict())
        pdp_acc = train_pdp_sgd_dpo_fixed(model_pdp, ref_model, private_loader, test_loader, tokenizer, device,
                                          args.max_grad_norm, args.lr, V_k, total_steps=total_steps, sigma=args.sigma)['overall_accuracy']
        torch.save(model_pdp.state_dict(), f"dpo_pdp_subset{i}_sigma{args.sigma}.pth")

        results.append({"subset": i, "gsd": gsd, "pdp_acc": pdp_acc, "dp_acc": dp_acc})

    # Print results
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    print("| Subset | GSD     | PDP Acc | DP Acc |")
    print("|--------|---------|---------|--------|")
    for r in results:
        print(f"| {r['subset']}      | {r['gsd']:.3f}   | {r['pdp_acc']:.3f}   | {dp_acc:.3f}  |")
    print("="*80)

    with open(f"results_{now_ts()}.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
