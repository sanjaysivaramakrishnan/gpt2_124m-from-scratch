"""
GPT-2 Training Script

Trains GPT-2 (124M) from scratch on tokenized data shards.

Features:
  - Distributed Data Parallel (DDP) multi-GPU support
  - Gradient accumulation for large effective batch sizes
  - Mixed-precision training (bfloat16)
  - Cosine learning-rate schedule with linear warmup
  - Checkpoint saving & resumption (including dataloader state)

Usage:
  Single GPU:
    python train.py

  Multi-GPU (DDP):
    torchrun --nproc_per_node=N train.py

  Override defaults:
    python train.py --max_steps 5000 --batch_size 32 --lr 3e-4
    python train.py --resume log/model_10000.pt
"""

import argparse
import math
import os
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

from model import GPT, GPTConfig
import config as cfg


# ── Helpers ──────────────────────────────────────────────────────────────

def load_tokens(filename):
    """Load a tokenized .npy shard and return a long tensor."""
    npt = np.load(filename).astype(np.int32)
    return torch.tensor(npt, dtype=torch.long)


class DataLoaderLite:
    """Lightweight data loader that streams from .npy shards."""

    def __init__(self, B, T, process_rank, num_processes, split, data_root, verbose=True):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes

        assert split in {"train", "val"}

        shards = sorted(s for s in os.listdir(data_root) if split in s)
        self.shards = [os.path.join(data_root, s) for s in shards]
        assert len(self.shards) > 0, f"No shards found for split: {split}"

        if verbose:
            print(f"Found {len(self.shards)} shards for split: {split}")

        self.reset()

    def reset(self):
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def _advance_shard(self):
        """Move to next shard (wraps around)."""
        self.current_shard = (self.current_shard + 1) % len(self.shards)
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        block = self.B * self.T

        # Ensure enough tokens are available
        while self.current_position + block + 1 > len(self.tokens):
            self._advance_shard()

        buf = self.tokens[self.current_position : self.current_position + block + 1]
        x = buf[:-1].reshape(self.B, self.T)
        y = buf[1:].reshape(self.B, self.T)

        self.current_position += self.B * self.T * self.num_processes
        return x, y


def get_lr(step, warmup_steps, max_steps, max_lr, min_lr):
    """Cosine learning-rate schedule with linear warmup."""
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step > max_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


# ── CLI ──────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train GPT-2 from scratch")
    p.add_argument("--batch_size",   type=int,   default=cfg.MICRO_BATCH_SIZE)
    p.add_argument("--seq_len",      type=int,   default=cfg.SEQ_LEN)
    p.add_argument("--max_steps",    type=int,   default=cfg.MAX_STEPS)
    p.add_argument("--lr",           type=float, default=cfg.MAX_LR)
    p.add_argument("--warmup_steps", type=int,   default=cfg.WARMUP_STEPS)
    p.add_argument("--weight_decay", type=float, default=cfg.WEIGHT_DECAY)
    p.add_argument("--grad_clip",    type=float, default=cfg.GRAD_CLIP)
    p.add_argument("--eval_interval",type=int,   default=cfg.EVAL_INTERVAL)
    p.add_argument("--data_dir",     type=str,   default=cfg.DATA_DIR)
    p.add_argument("--log_dir",      type=str,   default=cfg.LOG_DIR)
    p.add_argument("--resume",       type=str,   default=None, help="Path to checkpoint to resume from")
    p.add_argument("--no_compile",   action="store_true", help="Disable torch.compile")
    return p.parse_args()


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── DDP setup ────────────────────────────────────────────────────────
    ddp = int(os.environ.get("RANK", -1)) != -1

    if ddp:
        assert torch.cuda.is_available(), "DDP requires CUDA"
        init_process_group(backend="nccl")
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        device = f"cuda:{ddp_local_rank}"
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0
    else:
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        master_process = True
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if master_process:
        print(f"Running on {'GPU' if device != 'cpu' else 'CPU'} × {ddp_world_size}")

    device_type = "cuda" if "cuda" in device else "cpu"

    # ── Seed ─────────────────────────────────────────────────────────────
    seed = cfg.SEED + ddp_rank
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # ── Batch / gradient accumulation ────────────────────────────────────
    B, T = args.batch_size, args.seq_len
    total_batch_size = cfg.TOTAL_BATCH_SIZE
    assert total_batch_size % (B * T * ddp_world_size) == 0
    grad_accum_steps = total_batch_size // (B * T * ddp_world_size)

    if master_process:
        print(f"Total batch size: {total_batch_size:,} tokens")
        print(f"Gradient accumulation steps: {grad_accum_steps}")

    # ── Data ─────────────────────────────────────────────────────────────
    train_loader = DataLoaderLite(B, T, ddp_rank, ddp_world_size, "train", args.data_dir, verbose=master_process)
    val_loader   = DataLoaderLite(B, T, ddp_rank, ddp_world_size, "val",   args.data_dir, verbose=master_process)

    # ── Model ────────────────────────────────────────────────────────────
    torch.set_float32_matmul_precision("high")
    model = GPT(GPTConfig())
    model.to(device)

    if not args.no_compile:
        model = torch.compile(model)

    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])

    raw_model = model.module if ddp else model

    if master_process:
        print("Model loaded successfully")

    # ── Optimizer ────────────────────────────────────────────────────────
    max_lr = args.lr
    min_lr = max_lr * 0.1
    optimizer = raw_model.optimizer_configuration(args.weight_decay, max_lr, device, master_process)

    # ── Resume ───────────────────────────────────────────────────────────
    start_step = 0
    if args.resume and os.path.exists(args.resume):
        if master_process:
            print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        raw_model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt["step"] + 1
        train_loader.current_shard = ckpt["train_shard"]
        train_loader.tokens = load_tokens(train_loader.shards[train_loader.current_shard])
        train_loader.current_position = ckpt["train_position"]
        if master_process:
            print(f"Resuming training from step {start_step}")

    # ── Logging ──────────────────────────────────────────────────────────
    os.makedirs(args.log_dir, exist_ok=True)
    log_file = os.path.join(args.log_dir, "log.txt")

    # ── Training loop ────────────────────────────────────────────────────
    for step in range(start_step, args.max_steps):
        t0 = time.time()
        last_step = step == args.max_steps - 1

        # ── Validation ───────────────────────────────────────────────────
        if step % args.eval_interval == 0 or last_step:
            model.eval()
            val_loader.reset()
            with torch.no_grad():
                val_loss_accum = 0.0
                for _ in range(cfg.VAL_LOSS_STEPS):
                    x, y = val_loader.next_batch()
                    x, y = x.to(device), y.to(device)
                    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                        _, loss = model(x, y)
                    val_loss_accum += loss.detach() / cfg.VAL_LOSS_STEPS
                if ddp:
                    dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
                if master_process:
                    print(f"validation loss: {val_loss_accum.item():.4f}")
                    with open(log_file, "a") as f:
                        f.write(f"{step} val {val_loss_accum.item():.4f}\n")

            # Save checkpoint
            ckpt_path = os.path.join(args.log_dir, f"model_{step:05d}.pt")
            ckpt = {
                "model": raw_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": raw_model.config.__dict__,
                "step": step,
                "train_shard": train_loader.current_shard,
                "train_position": train_loader.current_position,
            }
            if master_process:
                torch.save(ckpt, ckpt_path)

        # ── Training step ────────────────────────────────────────────────
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_accum = 0.0

        for micro_step in range(grad_accum_steps):
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)
            if ddp:
                model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                _, loss = model(x, y)
                loss = loss / grad_accum_steps
                loss_accum += loss.detach()
                loss.backward()

        if ddp:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

        norm = torch.nn.utils.clip_grad_norm_(raw_model.parameters(), args.grad_clip)

        lr = get_lr(step, args.warmup_steps, args.max_steps, max_lr, min_lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr
        optimizer.step()

        if "cuda" in device:
            torch.cuda.synchronize()

        dt = time.time() - t0
        tok_per_sec = B * T * grad_accum_steps * ddp_world_size / dt
        if master_process:
            print(
                f"step {step:4d} | loss: {loss_accum.item():.6f} | "
                f"lr {lr:.4e} | norm: {norm:.4f} | "
                f"dt: {dt*1000:.2f}ms | tok/sec: {tok_per_sec:.2f}"
            )

    if ddp:
        destroy_process_group()


if __name__ == "__main__":
    main()