#!/usr/bin/env python3
"""Train a diffusion policy conditioned on frozen ObjectFlow predictions or GT masks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from phase2_core import DDPMSchedule
from phase2_core import DualViewActionDataset
from phase2_core import FrozenObjectFlowAdapter
from phase2_core import InstructionTokenizer
from phase2_core import NormalizationStats
from phase2_core import compute_normalization_stats
from phase2_core import load_trainable_state_dict
from phase2_core import normalize_tensor
from phase2_core import parse_episode_spec
from phase2_core import policy_from_config
from phase2_core import set_seed
from phase2_core import trainable_state_dict
import torch
import torch.nn.functional as functional
from torch.utils.data import DataLoader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, help="Folder containing epi_xxx robot recordings.")
    parser.add_argument("--flow-root", default="../object_flow_project/data/slide_test_2_obj")
    parser.add_argument("--output-dir", default="runs/flow_conditioned")
    parser.add_argument("--train-episodes", default="000-013", help="Episode IDs/ranges, e.g. 000-013,020")
    parser.add_argument("--val-episodes", default="014-016")
    parser.add_argument("--state-key", default="pose")
    parser.add_argument("--action-key", default="action")
    parser.add_argument("--obs-horizon", type=int, default=3)
    parser.add_argument("--action-horizon", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--expected-action-dim", type=int, default=7)

    parser.add_argument("--flow-source", choices=("predicted", "ground_truth"), default="predicted")
    parser.add_argument(
        "--object-flow-checkpoint",
        default="../object_flow_project/runs/slide_test_2_obj_paperlike_e1000_bs32/best.pt",
    )
    parser.add_argument("--object-flow-project", default="../object_flow_project")
    parser.add_argument("--object-flow-vocab", default=None)
    parser.add_argument("--flow-token-layers", type=int, default=2)

    parser.add_argument("--vision-encoder", choices=("dinov2", "small_cnn"), default="dinov2")
    parser.add_argument("--dinov2-model", default="dinov2_vits14")
    parser.add_argument("--language-encoder", choices=("bert", "hash"), default="bert")
    parser.add_argument("--bert-model", default="bert-base-uncased")
    parser.add_argument("--max-text-len", type=int, default=32)
    parser.add_argument("--condition-dim", type=int, default=256)
    parser.add_argument("--observation-token-layers", type=int, default=2)
    parser.add_argument("--base-channels", type=int, default=128)

    parser.add_argument("--diffusion-steps", type=int, default=100)
    parser.add_argument("--beta-start", type=float, default=1e-4)
    parser.add_argument("--beta-end", type=float, default=2e-2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-val-batches", type=int, default=0)
    return parser.parse_args()


def resolve_device(value: str) -> torch.device:
    if value != "auto":
        return torch.device(value)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def move_batch(batch: dict[str, Any], device: torch.device, stats: NormalizationStats) -> dict[str, Any]:
    moved = {
        "agent_rgb": batch["agent_rgb"].to(device, non_blocking=True),
        "wrist_rgb": batch["wrist_rgb"].to(device, non_blocking=True),
        "robot_state": normalize_tensor(
            batch["robot_state"].to(device, non_blocking=True), stats.state_mean, stats.state_std
        ),
        "action": normalize_tensor(batch["action"].to(device, non_blocking=True), stats.action_mean, stats.action_std),
        "language": batch["language"],
    }
    if "flow_mask" in batch:
        moved["flow_mask"] = batch["flow_mask"].to(device, non_blocking=True)
    return moved


def select_flow_masks(
    batch: dict[str, Any],
    source: str,
    adapter: FrozenObjectFlowAdapter | None,
) -> torch.Tensor:
    if source == "ground_truth":
        if "flow_mask" not in batch:
            raise KeyError("Ground-truth flow masks were requested but are absent from the batch.")
        return batch["flow_mask"]
    if adapter is None:
        raise RuntimeError("Predicted flow source requires an ObjectFlow adapter.")
    return adapter.predict(batch["agent_rgb"], batch["language"])


def run_epoch(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    tokenizer: InstructionTokenizer,
    schedule: DDPMSchedule,
    stats: NormalizationStats,
    device: torch.device,
    flow_source: str,
    flow_adapter: FrozenObjectFlowAdapter | None,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler,
    grad_clip: float,
    amp_enabled: bool,
    max_batches: int,
) -> float:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_samples = 0
    context = torch.enable_grad if training else torch.no_grad
    with context():
        for batch_index, raw_batch in enumerate(loader):
            if max_batches > 0 and batch_index >= max_batches:
                break
            batch = move_batch(raw_batch, device, stats)
            flow_masks = select_flow_masks(batch, flow_source, flow_adapter)
            text = tokenizer(batch["language"], device)
            timesteps = torch.randint(schedule.num_steps, (len(batch["action"]),), device=device)
            noise = torch.randn_like(batch["action"])
            noisy_actions = schedule.add_noise(batch["action"], noise, timesteps)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                condition = model.encode_condition(
                    batch["agent_rgb"],
                    batch["wrist_rgb"],
                    batch["robot_state"],
                    text["input_ids"],
                    text["attention_mask"],
                    flow_masks,
                )
                predicted_noise = model(noisy_actions, timesteps, condition)
                loss = functional.mse_loss(predicted_noise, noise)
            if training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            total_loss += float(loss.detach()) * len(batch["action"])
            total_samples += len(batch["action"])
    if total_samples == 0:
        raise RuntimeError("No batches were processed.")
    return total_loss / total_samples


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    require_flow = args.flow_source == "ground_truth"

    train_dataset = DualViewActionDataset(
        args.data_root,
        parse_episode_spec(args.train_episodes),
        obs_horizon=args.obs_horizon,
        action_horizon=args.action_horizon,
        image_size=args.image_size,
        state_key=args.state_key,
        action_key=args.action_key,
        flow_root=args.flow_root if require_flow else None,
        require_flow_masks=require_flow,
    )
    val_dataset = DualViewActionDataset(
        args.data_root,
        parse_episode_spec(args.val_episodes),
        obs_horizon=args.obs_horizon,
        action_horizon=args.action_horizon,
        image_size=args.image_size,
        state_key=args.state_key,
        action_key=args.action_key,
        flow_root=args.flow_root if require_flow else None,
        require_flow_masks=require_flow,
    )
    if train_dataset.action_dim != args.expected_action_dim:
        raise ValueError(
            f"Dataset action_dim={train_dataset.action_dim}, expected {args.expected_action_dim}. "
            "Check the action representation before training."
        )
    if val_dataset.state_dim != train_dataset.state_dim or val_dataset.action_dim != train_dataset.action_dim:
        raise ValueError("Train/validation state or action dimensions do not match.")

    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.num_workers > 0,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_options)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_options)

    config = vars(args).copy()
    config.update(
        {
            "policy_type": "flow_conditioned",
            "state_dim": train_dataset.state_dim,
            "action_dim": train_dataset.action_dim,
        }
    )
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    model = policy_from_config(config, use_flow=True).to(device)
    tokenizer = InstructionTokenizer(args.language_encoder, args.bert_model, args.max_text_len)
    schedule = DDPMSchedule(args.diffusion_steps, args.beta_start, args.beta_end).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    amp_enabled = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    flow_adapter = None
    if args.flow_source == "predicted":
        flow_adapter = FrozenObjectFlowAdapter(
            args.object_flow_checkpoint,
            args.object_flow_project,
            device,
            vocab_path=args.object_flow_vocab,
        )
        if int(flow_adapter.args["obs_horizon"]) != args.obs_horizon:
            raise ValueError("ObjectFlow and policy observation horizons do not match.")
        if int(flow_adapter.args["flow_horizon"]) != args.action_horizon:
            raise ValueError("ObjectFlow horizon and policy action horizon do not match.")

    stats = compute_normalization_stats(train_dataset)
    start_epoch = 0
    best_val = float("inf")
    if args.resume is not None:
        saved = torch.load(args.resume, map_location="cpu", weights_only=False)
        if saved.get("policy_type") != "flow_conditioned":
            raise ValueError(f"Not a flow-conditioned checkpoint: {args.resume}")
        load_trainable_state_dict(model, saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        stats = NormalizationStats.from_dict(saved["normalization"])
        start_epoch = int(saved["epoch"])
        best_val = float(saved.get("best_val", float("inf")))

    for epoch in range(start_epoch, args.epochs):
        train_loss = run_epoch(
            model=model,
            loader=train_loader,
            tokenizer=tokenizer,
            schedule=schedule,
            stats=stats,
            device=device,
            flow_source=args.flow_source,
            flow_adapter=flow_adapter,
            optimizer=optimizer,
            scaler=scaler,
            grad_clip=args.grad_clip,
            amp_enabled=amp_enabled,
            max_batches=args.max_train_batches,
        )
        val_loss = run_epoch(
            model=model,
            loader=val_loader,
            tokenizer=tokenizer,
            schedule=schedule,
            stats=stats,
            device=device,
            flow_source=args.flow_source,
            flow_adapter=flow_adapter,
            optimizer=None,
            scaler=scaler,
            grad_clip=args.grad_clip,
            amp_enabled=amp_enabled,
            max_batches=args.max_val_batches,
        )
        print(f"epoch={epoch + 1}/{args.epochs} train_noise_mse={train_loss:.6f} val_noise_mse={val_loss:.6f}")
        checkpoint = {
            "format_version": 1,
            "policy_type": "flow_conditioned",
            "epoch": epoch + 1,
            "best_val": min(best_val, val_loss),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "config": config,
            "normalization": stats.to_dict(),
            "model": trainable_state_dict(model),
            "optimizer": optimizer.state_dict(),
            "object_flow_checkpoint": args.object_flow_checkpoint if args.flow_source == "predicted" else None,
        }
        torch.save(checkpoint, output_dir / "latest.pt")
        if args.save_every > 0 and (epoch + 1) % args.save_every == 0:
            torch.save(checkpoint, output_dir / f"checkpoint_epoch_{epoch + 1:04d}.pt")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(checkpoint, output_dir / "best.pt")


if __name__ == "__main__":
    main()
