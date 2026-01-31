#!/usr/bin/env python3
"""
MMLU-Pro – SFT + DPO (DP-SGD & Fixed-PDP-SGD) with
* epoch-wise overall accuracy logging
* category-wise accuracy per public subset (no averaging)
* model checkpoint after every epoch
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
def setup_logging(log_file="mmlu_pro_category.log"):
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
def now_ts(): return datetime.now().strftime("%Y%m%dT%H%M%S")
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
        logits = torch.clamp(self.beta * ((pc - pr) - (rc - rr)), max=100)
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
# Evaluation – **category-wise**
# ============================================================================
@torch.no_grad()
def evaluate_category_wise(model, loader, tokenizer, device):
    model.eval()
    letters = [chr(ord('A')+i) for i in range(10)]
    completions = [f" {l}" for l in letters]
    token_ids = [tokenizer.encode(c, add_special_tokens=False)[0] if tokenizer.encode(c, add_special_tokens=False) else tokenizer.pad_token_id
                 for c in completions]
    token_ids = torch.tensor(token_ids, device=device)
    category_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    def get_prob(input_ids, attention_mask, token_id):
        if token_id == tokenizer.pad_token_id: return 0.0
        out = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = out.logits[:, -1, :]
        return F.softmax(logits, dim=-1)[0, token_id].item()
    for batch in loader:
        chosen_ids = batch[0].to(device) # prompt + chosen answer
        chosen_mask = batch[1].to(device)
        chosen_labels= batch[2].to(device)
        correct_idx = batch[6].to(device) # index of the correct letter (0-9)
        cat_ids = batch[7].to(device) # category index (int)
        for i in range(chosen_ids.size(0)):
            try:
                non_ignore = (chosen_labels[i] != -100).nonzero(as_tuple=True)[0]
                if len(non_ignore) == 0: continue
                prompt_len = non_ignore[0].item()
                input_ids = chosen_ids[i][:prompt_len].unsqueeze(0)
                attention_mask = chosen_mask[i][:prompt_len].unsqueeze(0)
                probs = [get_prob(input_ids, attention_mask, tid) for tid in token_ids]
                pred_idx = int(np.argmax(probs))
                true_idx = int(correct_idx[i].item())
                cat = int(cat_ids[i].item())
                category_stats[cat]["total"] += 1
                if pred_idx == true_idx:
                    category_stats[cat]["correct"] += 1
            except Exception as e:
                logging.warning(f"Eval error: {e}")
    acc_dict = {cat: float(stats["correct"]/max(stats["total"],1)) for cat, stats in category_stats.items()}
    overall = float(sum(s["correct"] for s in category_stats.values()) / max(sum(s["total"] for s in category_stats.values()),1))
    return acc_dict, overall
# -------------------------------------------------------------------------
# Add category ids to a DPO TensorDataset (8-th tensor)
# -------------------------------------------------------------------------
def add_category_to_dataset(dataset, category_list):
    tensors = [t for t in dataset.tensors]
    cat_tensor = torch.tensor(category_list, dtype=torch.long)
    tensors.append(cat_tensor)
    return TensorDataset(*tensors)
# ============================================================================
# DP-SGD SFT (epoch-wise accuracy + checkpoint)
# ============================================================================
def train_dp_sgd_sft(model, loader, test_loader, tokenizer, device,
                     max_grad_norm, lr, epochs=3, sigma=10.0, save_dir="checkpoints"):
    os.makedirs(save_dir, exist_ok=True)
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
                loss = model(input_ids=input_ids[i:i+1],
                             attention_mask=attention_mask[i:i+1],
                             labels=labels[i:i+1]).loss
                if loss is None or not torch.isfinite(loss): continue
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
        # ----- epoch evaluation -----
        _, overall = evaluate_category_wise(model, test_loader, tokenizer, device)
        logging.info(f"SFT DP-SGD Epoch {epoch+1}/{epochs} – Overall Acc = {overall:.4f}")
        # ----- checkpoint -----
        ckpt_path = os.path.join(save_dir, f"model_sft_dp_epoch{epoch+1}.pt")
        torch.save(model.state_dict(), ckpt_path)
        logging.info(f"Saved checkpoint: {ckpt_path}")
    return overall
# ============================================================================
# DP-SGD DPO (epoch-wise accuracy + checkpoint)
# ============================================================================
def train_dp_sgd_dpo(model, ref_model, loader, test_loader, tokenizer, device,
                     max_grad_norm, lr, epochs=3, sigma=10.0, save_dir="checkpoints"):
    os.makedirs(save_dir, exist_ok=True)
    optimizer = AdamW(trainable_params(model), lr=lr)
    dpo_loss_fn = DPOLoss(beta=0.1)
    tparams = trainable_params(model)
    param_sizes = [p.numel() for p in tparams]
    param_shapes = [p.shape for p in tparams]
    for epoch in range(epochs):
        model.train(); ref_model.eval()
        for batch in tqdm(loader, desc=f"DPO DP-SGD Epoch {epoch+1}"):
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
                    rc = get_batch_logps(ref_model(input_ids=c_ids[i:i+1], attention_mask=c_mask[i:i+1]).logits, c_lab[i:i+1])
                    rr = get_batch_logps(ref_model(input_ids=r_ids[i:i+1], attention_mask=r_mask[i:i+1]).logits, r_lab[i:i+1])
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
        # ----- epoch evaluation -----
        _, overall = evaluate_category_wise(model, test_loader, tokenizer, device)
        logging.info(f"DPO DP-SGD Epoch {epoch+1}/{epochs} – Overall Acc = {overall:.4f}")
        # ----- checkpoint -----
        ckpt_path = os.path.join(save_dir, f"model_dpo_dp_epoch{epoch+1}.pt")
        torch.save(model.state_dict(), ckpt_path)
        logging.info(f"Saved checkpoint: {ckpt_path}")
    return overall
# ============================================================================
# PDP-SGD SFT (fixed V_k) – epoch-wise accuracy + checkpoint
# ============================================================================
def train_pdp_sgd_sft_fixed(model, private_loader, test_loader, tokenizer, device,
                            max_grad_norm, lr, V_k, total_steps=1000, sigma=10.0,
                            save_dir="checkpoints"):
    os.makedirs(save_dir, exist_ok=True)
    optimizer = AdamW(trainable_params(model), lr=lr)
    tparams = trainable_params(model)
    param_sizes = [p.numel() for p in tparams]
    private_iter = iter(private_loader)
    step = 0
    pbar = tqdm(total=total_steps, desc="PDP-SGD SFT (fixed)")
    # compute steps per epoch
    steps_per_epoch = total_steps // 3 # 3 epochs by default
    epoch = 0
    while step < total_steps:
        step += 1; pbar.update(1)
        try: batch = next(private_iter)
        except StopIteration:
            private_iter = iter(private_loader)
            batch = next(private_iter)
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        labels = batch[2].to(device)
        per_sample_grads = []
        for i in range(input_ids.size(0)):
            model.zero_grad(set_to_none=True)
            loss = model(input_ids=input_ids[i:i+1],
                         attention_mask=attention_mask[i:i+1],
                         labels=labels[i:i+1]).loss
            if loss is None or not torch.isfinite(loss): continue
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
        # ----- epoch boundary -----
        if step % steps_per_epoch == 0:
            epoch += 1
            _, overall = evaluate_category_wise(model, test_loader, tokenizer, device)
            logging.info(f"PDP-SFT Epoch {epoch} – Overall Acc = {overall:.4f}")
            ckpt_path = os.path.join(save_dir, f"model_sft_pdp_epoch{epoch}.pt")
            torch.save(model.state_dict(), ckpt_path)
            logging.info(f"Saved checkpoint: {ckpt_path}")
    pbar.close()
    return evaluate_category_wise(model, test_loader, tokenizer, device)[1]
# ============================================================================
# PDP-SGD DPO (fixed V_k) – epoch-wise accuracy + checkpoint
# ============================================================================
def train_pdp_sgd_dpo_fixed(model, ref_model, private_loader, test_loader, tokenizer, device,
                            max_grad_norm, lr, V_k, total_steps=1000, sigma=10.0,
                            save_dir="checkpoints"):
    os.makedirs(save_dir, exist_ok=True)
    optimizer = AdamW(trainable_params(model), lr=lr)
    dpo_loss_fn = DPOLoss(beta=0.1)
    tparams = trainable_params(model)
    param_sizes = [p.numel() for p in tparams]
    private_iter = iter(private_loader)
    step = 0
    pbar = tqdm(total=total_steps, desc="PDP-SGD DPO (fixed)")
    steps_per_epoch = total_steps // 3
    epoch = 0
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
                rc = get_batch_logps(ref_model(input_ids=c_ids[i:i+1], attention_mask=c_mask[i:i+1]).logits, c_lab[i:i+1])
                rr = get_batch_logps(ref_model(input_ids=r_ids[i:i+1], attention_mask=r_mask[i:i+1]).logits, r_lab[i:i+1])
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
        if step % steps_per_epoch == 0:
            epoch += 1
            _, overall = evaluate_category_wise(model, test_loader, tokenizer, device)
            logging.info(f"PDP-DPO Epoch {epoch} – Overall Acc = {overall:.4f}")
            ckpt_path = os.path.join(save_dir, f"model_dpo_pdp_epoch{epoch}.pt")
            torch.save(model.state_dict(), ckpt_path)
            logging.info(f"Saved checkpoint: {ckpt_path}")
    pbar.close()
    return evaluate_category_wise(model, test_loader, tokenizer, device)[1]
# ============================================================================
# Dataset builders (SFT & DPO) – keep raw example for category
# ============================================================================
def build_sft_examples_from_hf(ds) -> list:
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
        examples.append({
            "prompt": format_mmlu_prompt(q, opts_shuf),
            "response": letters[correct_idx],
            "correct_idx": correct_idx,
            "raw": item
        })
    logging.info(f"SFT examples: {len(examples)} (skipped {dict(skipped)})")
    return examples
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
            "raw": item
        })
    logging.info(f"DPO examples: {len(examples)} (skipped {dict(skipped)})")
    return examples
def build_sft_dataset(examples, tokenizer, max_length=512):
    enc, eos = [], tokenizer.eos_token or "<|im_end|>"
    for ex in examples:
        full = tokenizer(ex["prompt"] + ex["response"] + eos,
                         truncation=True, padding="max_length",
                         max_length=max_length, return_tensors="pt")
        ids = full["input_ids"].squeeze(0)
        mask = full["attention_mask"].squeeze(0)
        labels = ids.clone()
        prompt_len = len(tokenizer(ex["prompt"], truncation=True, max_length=max_length)["input_ids"])
        labels[:prompt_len] = -100
        labels[ids == tokenizer.pad_token_id] = -100
        enc.append({"input_ids":ids, "attention_mask":mask, "labels":labels})
    if not enc: return None
    return TensorDataset(
        torch.stack([e["input_ids"] for e in enc]),
        torch.stack([e["attention_mask"] for e in enc]),
        torch.stack([e["labels"] for e in enc])
    )
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
# Gradient matrix & GSD
# ============================================================================
def compute_gradient_matrix_sft(model, loader, device, max_grad_norm):
    grads, tparams = [], trainable_params(model)
    for batch in loader:
        for i in range(batch[0].size(0)):
            model.zero_grad(set_to_none=True)
            loss = model(input_ids=batch[0][i:i+1].to(device),
                         attention_mask=batch[1][i:i+1].to(device),
                         labels=batch[2][i:i+1].to(device)).loss
            if loss is None: continue
            loss.backward()
            g = flatten_grads(tparams)
            g_norm = torch.norm(g).item()
            if g_norm > 0: g = g * min(1.0, max_grad_norm / g_norm)
            grads.append(g.cpu())
            model.zero_grad(set_to_none=True)
    return torch.stack(grads).numpy()
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
# ============================================================================
# MAIN
# ============================================================================
def main():
    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", default="./mmlu_pro_dataset", help="Local MMLU-Pro folder")
    parser.add_argument("--model_dir", default="./qwen_model", help="Local Qwen folder")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max_grad_norm",type=float, default=5.0)
    parser.add_argument("--sigma", type=float, default=10.0)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", default="checkpoints", help="Directory for epoch checkpoints")
    args = parser.parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    # ------------------- Load & shuffle once -------------------
    ds = load_from_disk(args.dataset_dir) # official test split
    N = len(ds)
    logging.info(f"Loaded MMLU-Pro test split: {N} examples")
    rng = np.random.default_rng(args.seed)
    shuffled_idx = rng.permutation(N).tolist()
    ds = ds.select(shuffled_idx)
    # ------------------- Deterministic split -------------------
    test_N = N // 16
    private_N = 10 * test_N
    public_N = 5 * test_N
    private = ds.select(range(private_N))
    public = ds.select(range(private_N, private_N + public_N))
    test = ds.select(range(private_N + public_N, private_N + public_N + test_N))
    # ------------------- Tokenizer -------------------
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    # ------------------- Build datasets -------------------
    private_sft_ex = build_sft_examples_from_hf(private)
    private_dpo_ex = build_dpo_examples_from_hf(private)
    test_dpo_ex = build_dpo_examples_from_hf(test)
    private_sft_ds = build_sft_dataset(private_sft_ex, tokenizer, args.max_length)
    private_dpo_ds = build_dpo_dataset(private_dpo_ex, tokenizer, args.max_length)
    test_dpo_ds = build_dpo_dataset(test_dpo_ex, tokenizer, args.max_length)
    if any(x is None for x in [private_sft_ds, private_dpo_ds, test_dpo_ds]):
        raise RuntimeError("Failed to build one of the datasets")
    # ---- add **numeric** category ids to the test loader ----
    cat_to_id = {cat: idx for idx, cat in enumerate(sorted(set(ex["raw"]["category"] for ex in test_dpo_ex)))}
    id_to_cat = {v: k for k, v in cat_to_id.items()}
    test_categories = [cat_to_id[ex["raw"]["category"]] for ex in test_dpo_ex]
    test_dpo_ds = add_category_to_dataset(test_dpo_ds, test_categories)
    num_batches = len(private_sft_ds) // args.batch_size
    total_steps = args.epochs * num_batches
    private_sft_loader = DataLoader(private_sft_ds,
                                    batch_sampler=FixedBatchSampler(private_sft_ds, args.batch_size))
    private_dpo_loader = DataLoader(private_dpo_ds,
                                    batch_sampler=FixedBatchSampler(private_dpo_ds, args.batch_size))
    test_loader = DataLoader(test_dpo_ds,
                             batch_sampler=FixedBatchSampler(test_dpo_ds, args.batch_size, drop_last=False))
    # ------------------- Base model -------------------
    init_model = build_qwen2_lora(args.model_dir).to(device)
    # ==================== DP-SGD pipeline ====================
    model_sft_dp = build_qwen2_lora(args.model_dir).to(device)
    model_sft_dp.load_state_dict(init_model.state_dict())
    sft_dp_acc = train_dp_sgd_sft(model_sft_dp, private_sft_loader, test_loader,
                                  tokenizer, device, args.max_grad_norm, args.lr,
                                  epochs=args.epochs, sigma=args.sigma, save_dir=args.save_dir)
    ref_dp = build_qwen2_lora(args.model_dir).to(device)
    ref_dp.load_state_dict(model_sft_dp.state_dict())
    ref_dp.eval()
    model_dpo_dp = build_qwen2_lora(args.model_dir).to(device)
    model_dpo_dp.load_state_dict(model_sft_dp.state_dict())
    dpo_dp_acc = train_dp_sgd_dpo(model_dpo_dp, ref_dp, private_dpo_loader, test_loader,
                                  tokenizer, device, args.max_grad_norm, args.lr,
                                  epochs=args.epochs, sigma=args.sigma, save_dir=args.save_dir)
    # ==================== PDP-SGD pipeline (5 public subsets) ====================
    results = []
    num_public_subsets = 5
    subset_size = len(public) // num_public_subsets
    for i in range(num_public_subsets):
        start = i * subset_size
        end = (i + 1) * subset_size if i < num_public_subsets - 1 else len(public)
        subset = public.select(range(start, end))
        name = f"subset_{i}"
        sft_ex = build_sft_examples_from_hf(subset)
        dpo_ex = build_dpo_examples_from_hf(subset)
        sft_ds = build_sft_dataset(sft_ex, tokenizer, args.max_length)
        dpo_ds = build_dpo_dataset(dpo_ex, tokenizer, args.max_length)
        if sft_ds is None or dpo_ds is None: continue
        public_sft_loader = DataLoader(sft_ds, batch_sampler=FixedBatchSampler(sft_ds, args.batch_size))
        public_dpo_loader = DataLoader(dpo_ds, batch_sampler=FixedBatchSampler(dpo_ds, args.batch_size))
        # ----- GSD -----
        G_priv_sft = compute_gradient_matrix_sft(init_model, private_sft_loader, device, args.max_grad_norm)
        G_pub_sft = compute_gradient_matrix_sft(init_model, public_sft_loader, device, args.max_grad_norm)
        sft_gsd = gradient_subspace_distance(G_priv_sft, G_pub_sft, args.k)
        G_priv_dpo = compute_gradient_matrix_dpo(model_sft_dp, ref_dp, private_dpo_loader, device, args.max_grad_norm)
        G_pub_dpo = compute_gradient_matrix_dpo(model_sft_dp, ref_dp, public_dpo_loader, device, args.max_grad_norm)
        dpo_gsd = gradient_subspace_distance(G_priv_dpo, G_pub_dpo, args.k)
        # ----- PDP-SFT -----
        _, _, Vt_sft = randomized_svd(G_pub_sft, n_components=args.k, random_state=42)
        V_k_sft = torch.from_numpy(Vt_sft.T).float().to(device)
        model_sft_pdp = build_qwen2_lora(args.model_dir).to(device)
        model_sft_pdp.load_state_dict(init_model.state_dict())
        sft_pdp_acc = train_pdp_sgd_sft_fixed(
            model=model_sft_pdp, private_loader=private_sft_loader,
            test_loader=test_loader, tokenizer=tokenizer, device=device,
            max_grad_norm=args.max_grad_norm, lr=args.lr,
            V_k=V_k_sft, total_steps=total_steps, sigma=args.sigma, save_dir=args.save_dir)
        ref_pdp = build_qwen2_lora(args.model_dir).to(device)
        ref_pdp.load_state_dict(model_sft_pdp.state_dict())
        ref_pdp.eval()
        # ----- PDP-DPO -----
        G_pub_dpo_pdp = compute_gradient_matrix_dpo(model_sft_pdp, ref_pdp, public_dpo_loader, device, args.max_grad_norm)
        _, _, Vt_dpo = randomized_svd(G_pub_dpo_pdp, n_components=args.k, random_state=42)
        V_k_dpo = torch.from_numpy(Vt_dpo.T).float().to(device)
        model_dpo_pdp = build_qwen2_lora(args.model_dir).to(device)
        model_dpo_pdp.load_state_dict(model_sft_pdp.state_dict())
        dpo_pdp_acc = train_pdp_sgd_dpo_fixed(
            model=model_dpo_pdp, ref_model=ref_pdp,
            private_loader=private_dpo_loader, test_loader=test_loader,
            tokenizer=tokenizer, device=device,
            max_grad_norm=args.max_grad_norm, lr=args.lr,
            V_k=V_k_dpo, total_steps=total_steps, sigma=args.sigma, save_dir=args.save_dir)
        # ----- Final category-wise numbers (per subset) -----
        cat_sft_dp, _ = evaluate_category_wise(model_sft_dp, test_loader, tokenizer, device)
        cat_dpo_dp, _ = evaluate_category_wise(model_dpo_dp, test_loader, tokenizer, device)
        cat_sft_pdp, _ = evaluate_category_wise(model_sft_pdp, test_loader, tokenizer, device)
        cat_dpo_pdp, _ = evaluate_category_wise(model_dpo_pdp, test_loader, tokenizer, device)
        results.append({
            "subset": name,
            "sft_gsd": sft_gsd,
            "dpo_gsd": dpo_gsd,
            "sft_dp_overall": sft_dp_acc,
            "dpo_dp_overall": dpo_dp_acc,
            "sft_pdp_overall": sft_pdp_acc,
            "dpo_pdp_overall": dpo_pdp_acc,
            "cat_sft_dp": cat_sft_dp,
            "cat_dpo_dp": cat_dpo_dp,
            "cat_sft_pdp": cat_sft_pdp,
            "cat_dpo_pdp": cat_dpo_pdp,
            "id_to_cat": id_to_cat
        })
    # ==================== Per-subset category-wise tables (no averaging) ====================
    print("\n" + "="*150)
    print("PER-SUBSET CATEGORY-WISE ACCURACY")
    print("="*150)
    for r in results:
        print(f"\nSubset: {r['subset']}")
        print("-" * 80)
        all_cat_ids = sorted(r["cat_sft_dp"].keys())
        header = "| Category | DP-SFT | PDP-SFT | DP-DPO | PDP-DPO |"
        print(header)
        sep = "|----------|---------|---------|---------|---------|"
        print(sep)
        for cid in all_cat_ids:
            cat_name = r["id_to_cat"].get(cid, str(cid))[:10]  # Truncate if too long
            row = f"| {cat_name:<8} | {r['cat_sft_dp'][cid]:>6.2%} | {r['cat_sft_pdp'][cid]:>6.2%} | {r['cat_dpo_dp'][cid]:>6.2%} | {r['cat_dpo_pdp'][cid]:>6.2%} |"
            print(row)
        print(sep)
        overall_header = "| Overall  |"
        overall_row = f"|          | {r['sft_dp_overall']:.2%} | {r['sft_pdp_overall']:.2%} | {r['dpo_dp_overall']:.2%} | {r['dpo_pdp_overall']:.2%} |"
        print(overall_header)
        print(overall_row)
        print("-" * 80)
    print("="*150)
    # Save full per-category dicts
    with open(f"category_results_{now_ts()}.json", "w") as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if hasattr(x, 'item') else x)

if __name__ == "__main__":
    main()
