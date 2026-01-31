#!/usr/bin/env python3
"""
RewardBench2 - Customized: 20%, 30%, 40% private data per public subset
Local model & dataset + save SFT & SFT+DPO models
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
def setup_logging(log_file="training_fixed_pdp_1.log"):
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
# Evaluation (unchanged)
# -----------------------
# ----------------------------------------------------------------------
# 7. Evaluation – USE correct_idx safely (same as second script)
# ----------------------------------------------------------------------
@torch.no_grad()
def evaluate_rewardbench_accuracy(model, loader, tokenizer, device, sorted_domains):
    model.eval()
    correct_by_domain = defaultdict(int)
    total_by_domain = defaultdict(int)
    letters = "ABCD"
    token_ids = [tokenizer(l, add_special_tokens=False)["input_ids"][0] for l in letters]
    token_ids = torch.tensor(token_ids, device=device)

    def get_prob(model, input_ids, attention_mask, token_id):
        if token_id == tokenizer.pad_token_id:
            return 0.0
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[:, -1, :]
        return F.softmax(logits, dim=-1)[0, token_id].item()

    for batch in loader:
        input_ids = batch[0].to(device)
        mask = batch[1].to(device)
        labels = batch[2].to(device)
        domain_idx = batch[3].to(device)
        # SFT loader now has 5 tensors → correct_idx is batch[4]
        correct_idx_batch = batch[4].to(device) if len(batch) >= 5 else torch.zeros(input_ids.size(0), dtype=torch.long, device=device)

        for i in range(input_ids.size(0)):
            labels_i = labels[i]
            non_ignore = (labels_i != -100).nonzero(as_tuple=True)[0]
            if len(non_ignore) == 0: continue
            prompt_len = non_ignore[0].item()
            prompt_ids = input_ids[i, :prompt_len].unsqueeze(0)
            prompt_mask = mask[i, :prompt_len].unsqueeze(0)
            probs = [get_prob(model, prompt_ids, prompt_mask, tid) for tid in token_ids]
            pred = np.argmax(probs)
            correct_idx_i = correct_idx_batch[i].item()
            dom = sorted_domains[domain_idx[i].item()]
            total_by_domain[dom] += 1
            if pred == correct_idx_i:
                correct_by_domain[dom] += 1

    acc = {d: correct_by_domain[d] / max(total_by_domain[d], 1) for d in total_by_domain}
    overall = sum(acc.values()) / len(acc) if acc else 0.0
    return {"domain_accuracies": acc, "overall_accuracy": overall}
# -----------------------
# Training Functions (unchanged, except for saving)
# -----------------------
def train_dp_sgd_sft(model, loader, test_loader, tokenizer, device, max_grad_norm, lr, epochs=3, sigma=10.0):
    optimizer = AdamW(trainable_params(model), lr=lr)
    tparams = trainable_params(model)
    param_sizes = [p.numel() for p in tparams]
    param_shapes = [p.shape for p in tparams]
    for epoch in range(epochs):
        model.train()
        for batch in tqdm(loader, desc=f"SFT DP-SGD Epoch {epoch+1}"):
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            labels = batch[2].to(device)
            per_sample_grads = []
            for i in range(input_ids.size(0)):
                model.zero_grad(set_to_none=True)
                loss = model(input_ids=input_ids[i:i+1], attention_mask=attention_mask[i:i+1], labels=labels[i:i+1]).loss
                if loss is None or not torch.isfinite(loss): continue
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
        logging.info(f"SFT DP-SGD E{epoch+1}: Acc={acc['overall_accuracy']:.4f}")
    return acc

def train_pdp_sgd_sft_fixed(model, private_loader, test_loader, tokenizer, device,
                            max_grad_norm, lr, V_k, total_steps=1000, sigma=10.0):
    optimizer = AdamW(trainable_params(model), lr=lr)
    tparams = trainable_params(model)
    param_sizes = [p.numel() for p in tparams]
    private_iter = iter(private_loader)
    step = 0
    pbar = tqdm(total=total_steps, desc="PDP-SGD SFT (fixed)")
    while step < total_steps:
        step += 1
        pbar.update(1)
        try:
            batch = next(private_iter)
        except StopIteration:
            private_iter = iter(private_loader)
            batch = next(private_iter)
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        labels = batch[2].to(device)
        per_sample_grads = []
        for i in range(input_ids.size(0)):
            model.zero_grad(set_to_none=True)
            loss = model(input_ids=input_ids[i:i+1], attention_mask=attention_mask[i:i+1], labels=labels[i:i+1]).loss
            if loss is None or not torch.isfinite(loss): continue
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
        g_tilde = V_k @ (V_k.T @ g_noisy)
        idx = 0
        for p, sz in zip(tparams, param_sizes):
            p.grad = g_tilde[idx:idx+sz].view(p.shape)
            idx += sz
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if step % 100 == 0:
            acc = evaluate_rewardbench_accuracy(model, test_loader, tokenizer, device)
            logging.info(f"PDP-SGD SFT Step {step}: Acc={acc['overall_accuracy']:.4f}")
    pbar.close()
    return evaluate_rewardbench_accuracy(model, test_loader, tokenizer, device)

def train_pdp_sgd_dpo_fixed(model, ref_model, private_loader, test_loader, tokenizer, device,
                            max_grad_norm, lr, V_k, total_steps=1000, sigma=10.0):
    optimizer = AdamW(trainable_params(model), lr=lr)
    dpo_loss_fn = DPOLoss(beta=0.1)
    tparams = trainable_params(model)
    param_sizes = [p.numel() for p in tparams]
    private_iter = iter(private_loader)
    step = 0
    pbar = tqdm(total=total_steps, desc="PDP-SGD DPO (fixed)")
    while step < total_steps:
        step += 1
        pbar.update(1)
        try:
            batch = next(private_iter)
        except StopIteration:
            private_iter = iter(private_loader)
            batch = next(private_iter)
        chosen_ids = batch[0].to(device)
        chosen_mask = batch[1].to(device)
        chosen_labels = batch[2].to(device)
        rej_ids = batch[3].to(device)
        rej_mask = batch[4].to(device)
        rej_labels = batch[5].to(device)
        per_sample_grads = []
        for i in range(chosen_ids.size(0)):
            model.zero_grad(set_to_none=True)
            c_out = model(input_ids=chosen_ids[i:i+1], attention_mask=chosen_mask[i:i+1])
            r_out = model(input_ids=rej_ids[i:i+1], attention_mask=rej_mask[i:i+1])
            pc = get_batch_logps(c_out.logits, chosen_labels[i:i+1])
            pr = get_batch_logps(r_out.logits, rej_labels[i:i+1])
            with torch.no_grad():
                rc = get_batch_logps(ref_model(input_ids=chosen_ids[i:i+1], attention_mask=chosen_mask[i:i+1]).logits, chosen_labels[i:i+1])
                rr = get_batch_logps(ref_model(input_ids=rej_ids[i:i+1], attention_mask=rej_mask[i:i+1]).logits, rej_labels[i:i+1])
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
        g_tilde = V_k @ (V_k.T @ g_noisy)
        idx = 0
        for p, sz in zip(tparams, param_sizes):
            p.grad = g_tilde[idx:idx+sz].view(p.shape)
            idx += sz
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if step % 100 == 0:
            acc = evaluate_rewardbench_accuracy(model, test_loader, tokenizer, device)
            logging.info(f"PDP-SGD DPO Step {step}: Acc={acc['overall_accuracy']:.4f}")
    pbar.close()
    return evaluate_rewardbench_accuracy(model, test_loader, tokenizer, device)

def train_dp_sgd_dpo(model, ref_model, loader, test_loader, tokenizer, device, max_grad_norm, lr, epochs=3, sigma=10.0):
    optimizer = AdamW(trainable_params(model), lr=lr)
    dpo_loss_fn = DPOLoss(beta=0.1)
    tparams = trainable_params(model)
    param_sizes = [p.numel() for p in tparams]
    param_shapes = [p.shape for p in tparams]
    for epoch in range(epochs):
        model.train()
        ref_model.eval()
        for batch in tqdm(loader, desc=f"DPO DP-SGD Epoch {epoch+1}"):
            chosen_ids = batch[0].to(device)
            chosen_mask = batch[1].to(device)
            chosen_labels = batch[2].to(device)
            rej_ids = batch[3].to(device)
            rej_mask = batch[4].to(device)
            rej_labels = batch[5].to(device)
            per_sample_grads = []
            for i in range(chosen_ids.size(0)):
                model.zero_grad(set_to_none=True)
                c_out = model(input_ids=chosen_ids[i:i+1], attention_mask=chosen_mask[i:i+1])
                r_out = model(input_ids=rej_ids[i:i+1], attention_mask=rej_mask[i:i+1])
                pc = get_batch_logps(c_out.logits, chosen_labels[i:i+1])
                pr = get_batch_logps(r_out.logits, rej_labels[i:i+1])
                with torch.no_grad():
                    rc = get_batch_logps(ref_model(input_ids=chosen_ids[i:i+1], attention_mask=chosen_mask[i:i+1]).logits, chosen_labels[i:i+1])
                    rr = get_batch_logps(ref_model(input_ids=rej_ids[i:i+1], attention_mask=rej_mask[i:i+1]).logits, rej_labels[i:i+1])
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
        logging.info(f"DPO DP-SGD E{epoch+1}: Acc={acc['overall_accuracy']:.4f}")
    return acc

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
# Dataset Helpers (unchanged)
# -----------------------
def format_rewardbench_prompt(prompt: str, options: List[str]) -> str:
    letters = ['A', 'B', 'C', 'D']
    opt_text = "\n".join([f"{letter}. {opt}" for letter, opt in zip(letters, options)])
    return f"This is a multiple-choice question. Select the best completion by providing only the single capital letter (A, B, C, or D) corresponding to the correct answer. Prompt: {prompt}\nOptions:\n{opt_text}\nAnswer: "

# ... [build_sft_examples_from_hf, build_dpo_examples_from_hf, build_sft_dataset, build_dpo_dataset] unchanged ...

# -----------------------
# Gradient & GSD (unchanged)
# -----------------------
# ... [compute_gradient_matrix_sft, compute_gradient_matrix_dpo, gradient_subspace_distance] unchanged ...
def build_sft_examples_from_hf(ds, num_examples: int = None) -> Dict[str, List[Dict[str, Any]]]:
    domains = set(ds["subset"]) if "subset" in ds.features else set()
    examples_by_domain = {domain.lower(): [] for domain in domains}
    examples_by_domain["unknown"] = []
    skipped_counts = Counter()
    letters = ['A', 'B', 'C', 'D']
    for i, item in enumerate(ds):
        if num_examples is not None and i >= num_examples: break
        if not valid_example(item): 
            skipped_counts['invalid'] += 1
            continue
        prompt = item.get("prompt", "")
        chosen = item["chosen"]
        if isinstance(chosen, list):
            chosen = chosen[0].strip() if chosen and isinstance(chosen[0], str) else ""
        rejected = item.get("rejected", [])
        options = [chosen] + [opt.strip() for opt in rejected if isinstance(opt, str) and opt.strip()][:3]
        while len(options) < 4:
            options.append("Placeholder response")
            skipped_counts['padded_options'] += 1
        random.shuffle(options)
        try:
            correct_idx = options.index(chosen)
        except ValueError:
            skipped_counts['chosen_not_in_options'] += 1
            continue
        ans = letters[correct_idx]
        domain = item.get("subset", "unknown").lower()
        if domain not in examples_by_domain:
            domain = "unknown"
            skipped_counts['unknown_domain'] += 1
        prompt_text = format_rewardbench_prompt(prompt, options)
        examples_by_domain[domain].append({
            "prompt": prompt_text,
            "response": ans,
            "options": options,
            "correct_idx": correct_idx,
            "domain": domain,
            "id": item.get("id", f"example_{i}")
        })
    return examples_by_domain

def build_dpo_examples_from_hf(ds, num_examples: int = None) -> Dict[str, List[Dict[str, Any]]]:
    domains = set(ds["subset"]) if "subset" in ds.features else set()
    examples_by_domain = {domain.lower(): [] for domain in domains}
    examples_by_domain["unknown"] = []
    skipped_counts = Counter()
    letters = ['A', 'B', 'C', 'D']
    for i, item in enumerate(ds):
        if num_examples is not None and i >= num_examples: break
        if not valid_example(item): 
            skipped_counts['invalid'] += 1
            continue
        prompt = item.get("prompt", "")
        chosen = item["chosen"]
        if isinstance(chosen, list):
            chosen = chosen[0].strip() if chosen and isinstance(chosen[0], str) else ""
        rejected = item.get("rejected", [])
        options = [chosen] + [opt.strip() for opt in rejected if isinstance(opt, str) and opt.strip()][:3]
        while len(options) < 4:
            options.append("Placeholder response")
            skipped_counts['padded_options'] += 1
        random.shuffle(options)
        try:
            correct_idx = options.index(chosen)
        except ValueError:
            skipped_counts['chosen_not_in_options'] += 1
            continue
        ans = letters[correct_idx]
        rejected_options = [l for l in letters if l != ans]
        rejected_letter = random.choice(rejected_options)
        domain = item.get("subset", "unknown").lower()
        if domain not in examples_by_domain:
            domain = "unknown"
            skipped_counts['unknown_domain'] += 1
        prompt_text = format_rewardbench_prompt(prompt, options)
        examples_by_domain[domain].append({
            "prompt": prompt_text,
            "chosen": ans,
            "rejected": rejected_letter,
            "options": options,
            "correct_idx": correct_idx,
            "domain": domain,
            "id": item.get("id", f"example_{i}")
        })
    return examples_by_domain

def build_sft_dataset(examples_by_domain, tokenizer, max_length=512):
    enc = []
    domain_indices = []
    domain_map = {domain: idx for idx, domain in enumerate(sorted(examples_by_domain.keys()))}
    eos = tokenizer.eos_token or "<|im_end|>"
    for domain in examples_by_domain:
        for ex in examples_by_domain[domain]:
            prompt_text = ex["prompt"]
            response = ex["response"] + eos
            enc_full = tokenizer(prompt_text + response, truncation=True, padding="max_length", max_length=max_length, return_tensors="pt")
            input_ids = enc_full["input_ids"].squeeze(0)
            attention_mask = enc_full["attention_mask"].squeeze(0)
            labels = input_ids.clone()
            prompt_len = len(tokenizer(prompt_text, truncation=True, max_length=max_length)["input_ids"])
            labels[:prompt_len] = -100
            labels[input_ids == tokenizer.pad_token_id] = -100
            enc.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "domain": domain_map[domain]
            })
            domain_indices.append(domain_map[domain])
    if not enc:
        return None
    dataset = TensorDataset(
        torch.stack([e["input_ids"] for e in enc]),
        torch.stack([e["attention_mask"] for e in enc]),
        torch.stack([e["labels"] for e in enc]),
        torch.tensor(domain_indices, dtype=torch.long)
    )
    return dataset

def build_dpo_dataset(examples_by_domain, tokenizer, max_length=512):
    enc = []
    domain_indices = []
    domain_map = {domain: idx for idx, domain in enumerate(sorted(examples_by_domain.keys()))}
    eos = tokenizer.eos_token or "<|im_end|>"
    for domain in examples_by_domain:
        for ex in examples_by_domain[domain]:
            prompt_text = ex["prompt"]
            chosen = ex["chosen"] + eos
            rejected = ex["rejected"] + eos
            enc_chosen = tokenizer(prompt_text + chosen, truncation=True, padding="max_length", max_length=max_length, return_tensors="pt")
            enc_rejected = tokenizer(prompt_text + rejected, truncation=True, padding="max_length", max_length=max_length, return_tensors="pt")
            chosen_input_ids = enc_chosen["input_ids"].squeeze(0)
            chosen_attention_mask = enc_chosen["attention_mask"].squeeze(0)
            chosen_labels = chosen_input_ids.clone()
            prompt_len = len(tokenizer(prompt_text, truncation=True, max_length=max_length)["input_ids"])
            chosen_labels[:prompt_len] = -100
            chosen_labels[chosen_input_ids == tokenizer.pad_token_id] = -100
            rejected_input_ids = enc_rejected["input_ids"].squeeze(0)
            rejected_attention_mask = enc_rejected["attention_mask"].squeeze(0)
            rejected_labels = rejected_input_ids.clone()
            rejected_labels[:prompt_len] = -100
            rejected_labels[rejected_input_ids == tokenizer.pad_token_id] = -100
            enc.append({
                "chosen_input_ids": chosen_input_ids,
                "chosen_attention_mask": chosen_attention_mask,
                "chosen_labels": chosen_labels,
                "rejected_input_ids": rejected_input_ids,
                "rejected_attention_mask": rejected_attention_mask,
                "rejected_labels": rejected_labels,
                "correct_idx": torch.tensor(ex["correct_idx"], dtype=torch.long),
                "domain": domain_map[domain]
            })
            domain_indices.append(domain_map[domain])
    if not enc:
        return None
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
    return dataset

# -----------------------
# Gradient Matrix & GSD
# -----------------------
def compute_gradient_matrix_sft(model, loader, device, max_grad_norm):
    grads = []
    tparams = trainable_params(model)
    for batch in loader:
        for i in range(batch[0].size(0)):
            model.zero_grad(set_to_none=True)
            loss = model(input_ids=batch[0][i:i+1].to(device), attention_mask=batch[1][i:i+1].to(device), labels=batch[2][i:i+1].to(device)).loss
            if loss is None: continue
            loss.backward()
            g = flatten_grads(tparams)
            g_norm = torch.norm(g).item()
            if g_norm > 0:
                g = g * min(1.0, max_grad_norm / g_norm)
            grads.append(g.cpu())
            model.zero_grad(set_to_none=True)
    return torch.stack(grads).numpy()

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
            if loss is None: continue
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
def build_qwen2_lora(name, fp16=False):
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float16 if fp16 else torch.bfloat16, trust_remote_code=True)
    if model.config.pad_token_id is None:
        model.config.pad_token_id = model.config.eos_token_id
    cfg = LoraConfig(r=8, lora_alpha=8, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], lora_dropout=0.1, bias="none", task_type=TaskType.CAUSAL_LM)
    return get_peft_model(model, cfg)

def now_ts():
    return datetime.now().strftime("%Y%m%dT%H%M%S")

# -----------------------
# Main
# -----------------------
def main():
    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="./qwen_model", help="Local Qwen model folder")
    parser.add_argument("--dataset_dir", default="./datasets/reward_bench_dataset", help="Local Reward-Bench folder")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max_grad_norm", type=float, default=5.0)
    parser.add_argument("--sigma", type=float, default=10.0)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--max_length", type=int, default=5160)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load local dataset
    ds = load_from_disk(args.dataset_dir).shuffle(seed=args.seed)
    N = len(ds)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    init_model = build_qwen2_lora(args.model_dir).to(device)
    model_name = os.path.basename(args.model_dir).replace(".", "_")
    dataset_name = "rewardbench"
    results = []
    # Loop over public/private ratios x
    for x in [10, 20, 30, 40]:
        # Compute sizes to fill total N exactly
        r = x / 100.0
        total_ratio = 1 + 0.1 + 5 * r
        private_N = int(N / total_ratio)  # Base private size
        test_N = int(0.1 * private_N)
        public_subset_N = int(r * private_N)
        # Adjust last public_subset to avoid rounding loss (ensure sum == N)
        public_total_N = 5 * public_subset_N
        adjustment = N - (private_N + test_N + public_total_N)
        if adjustment != 0:
            public_subset_N += adjustment // 5  # Distribute evenly
            public_total_N = 5 * public_subset_N
            # Re-adjust private if needed to fit
            private_N = N - test_N - public_total_N
        logging.info(f"\n=== Public/Private {x}%: Private {private_N} | Each Public {public_subset_N} | Test {test_N} | Total {private_N + test_N + public_total_N} ===")
        # Select slices (from end: test, then public pool, then private)
        test_start = N - test_N
        public_start = test_start - public_total_N
        private_start = max(0, public_start - private_N)
        test = ds.select(range(test_start, N))
        public = ds.select(range(public_start, test_start))  # Full public pool
        private = ds.select(range(private_start, public_start))
        # Build test DPO loader
        test_dpo_ex = build_dpo_examples_from_hf(test)
        test_dpo_ds = build_dpo_dataset(test_dpo_ex, tokenizer, args.max_length)
        test_dpo_loader = DataLoader(test_dpo_ds, batch_sampler=FixedBatchSampler(test_dpo_ds, args.batch_size, drop_last=False))
        # Build private datasets/loaders
        private_sft_ex = build_sft_examples_from_hf(private)
        private_dpo_ex = build_dpo_examples_from_hf(private)
        private_sft_ds = build_sft_dataset(private_sft_ex, tokenizer, args.max_length)
        private_dpo_ds = build_dpo_dataset(private_dpo_ex, tokenizer, args.max_length)
        if any(x is None for x in [private_sft_ds, private_dpo_ds]):
            logging.warning(f"Skipping x={x} due to dataset error")
            continue
        num_batches = len(private_sft_ds) // args.batch_size
        total_steps = args.epochs * num_batches
        private_sft_loader = DataLoader(private_sft_ds, batch_sampler=FixedBatchSampler(private_sft_ds, args.batch_size))
        private_dpo_loader = DataLoader(private_dpo_ds, batch_sampler=FixedBatchSampler(private_dpo_ds, args.batch_size))
        # DP-SGD baseline (once per x)
        model_sft_dp = build_qwen2_lora(args.model_dir).to(device)
        model_sft_dp.load_state_dict(init_model.state_dict())
        sft_dp_acc = train_dp_sgd_sft(model_sft_dp, private_sft_loader, test_dpo_loader, tokenizer, device,
                                      args.max_grad_norm, args.lr, epochs=args.epochs, sigma=args.sigma)['overall_accuracy']
        ref_dp = build_qwen2_lora(args.model_dir).to(device)
        ref_dp.load_state_dict(model_sft_dp.state_dict())
        ref_dp.eval()
        model_dpo_dp = build_qwen2_lora(args.model_dir).to(device)
        model_dpo_dp.load_state_dict(model_sft_dp.state_dict())
        dpo_dp_acc = train_dp_sgd_dpo(model_dpo_dp, ref_dp, private_dpo_loader, test_dpo_loader, tokenizer, device,
                                      args.max_grad_norm, args.lr, epochs=args.epochs, sigma=args.sigma)['overall_accuracy']
        G_priv_sft = compute_gradient_matrix_sft(init_model, private_sft_loader, device, args.max_grad_norm)
        G_priv_dpo = compute_gradient_matrix_dpo(model_sft_dp, ref_dp, private_dpo_loader, device, args.max_grad_norm)
        # Per public subset
        num_public_subsets = 5
        for i in range(num_public_subsets):
            start = i * public_subset_N
            end = min((i + 1) * public_subset_N, len(public))
            subset = public.select(range(start, end))
            name = f"subset_{i}_x{x}"
            print(f"\n--- Running: {name} | Private: {len(private)}, Public: {len(subset)}, Test: {len(test)} ---")
            sft_ex = build_sft_examples_from_hf(subset)
            dpo_ex = build_dpo_examples_from_hf(subset)
            sft_ds = build_sft_dataset(sft_ex, tokenizer, args.max_length)
            dpo_ds = build_dpo_dataset(dpo_ex, tokenizer, args.max_length)
            if sft_ds is None or dpo_ds is None:
                continue
            public_sft_loader = DataLoader(sft_ds, batch_sampler=FixedBatchSampler(sft_ds, args.batch_size))
            public_dpo_loader = DataLoader(dpo_ds, batch_sampler=FixedBatchSampler(dpo_ds, args.batch_size))
            G_pub_sft = compute_gradient_matrix_sft(init_model, public_sft_loader, device, args.max_grad_norm)
            sft_gsd = gradient_subspace_distance(G_priv_sft, G_pub_sft, args.k)
            G_pub_dpo = compute_gradient_matrix_dpo(model_sft_dp, ref_dp, public_dpo_loader, device, args.max_grad_norm)
            dpo_gsd = gradient_subspace_distance(G_priv_dpo, G_pub_dpo, args.k)
            # PDP-SGD SFT
            _, _, Vt_sft = randomized_svd(G_pub_sft, n_components=args.k, random_state=42)
            V_k_sft = torch.from_numpy(Vt_sft.T).float().to(device)
            model_sft_pdp = build_qwen2_lora(args.model_dir).to(device)
            model_sft_pdp.load_state_dict(init_model.state_dict())
            sft_pdp_acc = train_pdp_sgd_sft_fixed(
                model=model_sft_pdp, private_loader=private_sft_loader, test_loader=test_dpo_loader,
                tokenizer=tokenizer, device=device, max_grad_norm=args.max_grad_norm,
                lr=args.lr, V_k=V_k_sft, total_steps=total_steps, sigma=args.sigma
            )['overall_accuracy']
            # Save SFT model
            sft_save_path = f"model_{model_name}_{dataset_name}_sigma{args.sigma}_x{x}_sft_1.pth"
            torch.save(model_sft_pdp.state_dict(), sft_save_path)
            logging.info(f"Saved SFT model: {sft_save_path}")
            ref_pdp = build_qwen2_lora(args.model_dir).to(device)
            ref_pdp.load_state_dict(model_sft_pdp.state_dict())
            ref_pdp.eval()
            # PDP-SGD DPO
            G_pub_dpo_pdp = compute_gradient_matrix_dpo(model_sft_pdp, ref_pdp, public_dpo_loader, device, args.max_grad_norm)
            _, _, Vt_dpo = randomized_svd(G_pub_dpo_pdp, n_components=args.k, random_state=42)
            V_k_dpo = torch.from_numpy(Vt_dpo.T).float().to(device)
            model_dpo_pdp = build_qwen2_lora(args.model_dir).to(device)
            model_dpo_pdp.load_state_dict(model_sft_pdp.state_dict())
            dpo_pdp_acc = train_pdp_sgd_dpo_fixed(
                model=model_dpo_pdp, ref_model=ref_pdp, private_loader=private_dpo_loader,
                test_loader=test_dpo_loader, tokenizer=tokenizer, device=device,
                max_grad_norm=args.max_grad_norm, lr=args.lr, V_k=V_k_dpo,
                total_steps=total_steps, sigma=args.sigma
            )['overall_accuracy']
            # Save SFT+DPO model
            dpo_save_path = f"model_{model_name}_{dataset_name}_sigma{args.sigma}_x{x}_1.pth"
            torch.save(model_dpo_pdp.state_dict(), dpo_save_path)
            logging.info(f"Saved SFT+DPO model: {dpo_save_path}")
            results.append({
                "name": name,
                "public_ratio": x,
                "sft_gsd": sft_gsd,
                "dpo_gsd": dpo_gsd,
                "sft_pdp": sft_pdp_acc,
                "sft_dp": sft_dp_acc,
                "dpo_pdp": dpo_pdp_acc,
                "dpo_dp": dpo_dp_acc
            })
    # Final Table (now logged)
    logging.info("\n" + "="*100)
    logging.info("FINAL RESULTS SUMMARY")
    logging.info("="*100)
    logging.info("| Subset | Pub% | SFT GSD | DPO GSD | SFT PDP | SFT DP | DPO PDP | DPO DP |")
    logging.info("|--------|------|---------|---------|---------|--------|---------|--------|")
    for r in results:
        logging.info(f"| {r['name']:<12} | {r['public_ratio']:4} | {r['sft_gsd']:7.3f} | {r['dpo_gsd']:7.3f} | "
              f"{r['sft_pdp']:7.3f} | {r['sft_dp']:6.3f} | {r['dpo_pdp']:7.3f} | {r['dpo_dp']:6.3f} |")
    logging.info("="*100)
if __name__ == "__main__":
    main()
