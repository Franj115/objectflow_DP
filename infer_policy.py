#!/usr/bin/env python3
"""Offline inference for a baseline or flow-conditioned Phase-2 diffusion policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from phase2_core import DDPMSchedule
from phase2_core import FrozenObjectFlowAdapter
from phase2_core import InstructionTokenizer
from phase2_core import NormalizationStats
from phase2_core import denormalize_tensor
from phase2_core import instruction_for_episode
from phase2_core import load_single_observation
from phase2_core import load_trainable_state_dict
from phase2_core import normalize_tensor
from phase2_core import policy_from_config
from phase2_core import set_seed
import torch

ACTION_NAMES = ("delta_x", "delta_y", "delta_z", "delta_rx", "delta_ry", "delta_rz", "gripper")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="best.pt/latest.pt produced by either training script.")
    parser.add_argument(
        "--episode-dir", required=True, help="Raw epi_xxx folder containing two MP4 files and trajectory.pkl."
    )
    parser.add_argument("--frame-index", type=int, required=True, help="Current t; the script reads t-2:t by default.")
    parser.add_argument("--instruction", default=None, help="Overrides language_instruction.json for this sample.")
    parser.add_argument("--output-dir", default="outputs/inference")
    parser.add_argument("--object-flow-checkpoint", default=None)
    parser.add_argument("--object-flow-project", default=None)
    parser.add_argument("--object-flow-vocab", default=None)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def resolve_device(value: str) -> torch.device:
    if value != "auto":
        return torch.device(value)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_flow_masks(flow_masks: torch.Tensor, output_dir: Path) -> None:
    mask_dir = output_dir / "predicted_flow_masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    masks = flow_masks[0, :, 0].detach().cpu().numpy()
    for index, mask in enumerate(masks):
        cv2.imwrite(str(mask_dir / f"flow_{index:03d}.png"), np.clip(mask * 255.0, 0, 255).astype(np.uint8))


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    policy_type = saved.get("policy_type")
    if policy_type not in {"baseline", "flow_conditioned"}:
        raise ValueError(f"Unknown policy_type={policy_type!r} in {args.checkpoint}")
    config = dict(saved["config"])
    stats = NormalizationStats.from_dict(saved["normalization"])
    use_flow = policy_type == "flow_conditioned"

    model = policy_from_config(config, use_flow=use_flow).to(device)
    load_trainable_state_dict(model, saved["model"])
    model.eval()
    tokenizer = InstructionTokenizer(
        str(config["language_encoder"]), str(config["bert_model"]), int(config["max_text_len"])
    )
    schedule = DDPMSchedule(int(config["diffusion_steps"]), float(config["beta_start"]), float(config["beta_end"])).to(
        device
    )

    agent_rgb, wrist_rgb, robot_state = load_single_observation(
        args.episode_dir,
        frame_index=args.frame_index,
        obs_horizon=int(config["obs_horizon"]),
        image_size=int(config["image_size"]),
        state_key=str(config["state_key"]),
    )
    instruction = args.instruction if args.instruction is not None else instruction_for_episode(args.episode_dir)
    agent_rgb = agent_rgb.unsqueeze(0).to(device)
    wrist_rgb = wrist_rgb.unsqueeze(0).to(device)
    robot_state = normalize_tensor(robot_state.unsqueeze(0).to(device), stats.state_mean, stats.state_std)
    text = tokenizer([instruction], device)

    flow_masks = None
    if use_flow:
        flow_checkpoint = (
            args.object_flow_checkpoint or saved.get("object_flow_checkpoint") or config.get("object_flow_checkpoint")
        )
        project_root = args.object_flow_project or config.get("object_flow_project")
        if not flow_checkpoint or not project_root:
            raise ValueError(
                "Flow-conditioned inference needs --object-flow-checkpoint and --object-flow-project "
                "when those paths are absent from the policy checkpoint."
            )
        adapter = FrozenObjectFlowAdapter(
            flow_checkpoint,
            project_root,
            device,
            vocab_path=args.object_flow_vocab or config.get("object_flow_vocab"),
        )
        flow_masks = adapter.predict(agent_rgb, [instruction])

    with torch.inference_mode():
        condition = model.encode_condition(
            agent_rgb,
            wrist_rgb,
            robot_state,
            text["input_ids"],
            text["attention_mask"],
            flow_masks,
        )
        condition = condition.expand(args.num_samples, -1)
        normalized_actions = schedule.sample(
            model,
            condition,
            action_horizon=int(config["action_horizon"]),
            action_dim=int(config["action_dim"]),
        )
        actions = denormalize_tensor(normalized_actions, stats.action_mean, stats.action_std)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    actions_numpy = actions.detach().cpu().numpy()
    np.save(output_dir / "action_samples.npy", actions_numpy)
    action_names = (
        ACTION_NAMES
        if actions_numpy.shape[-1] == len(ACTION_NAMES)
        else tuple(f"action_{index}" for index in range(actions_numpy.shape[-1]))
    )
    samples_json = [
        [dict(zip(action_names, map(float, step), strict=True)) for step in sample] for sample in actions_numpy
    ]
    (output_dir / "action_samples.json").write_text(json.dumps(samples_json, indent=2), encoding="utf-8")
    metadata = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "policy_type": policy_type,
        "episode_dir": str(Path(args.episode_dir).resolve()),
        "frame_index": args.frame_index,
        "instruction": instruction,
        "num_samples": args.num_samples,
        "action_horizon": int(config["action_horizon"]),
        "action_dim": int(config["action_dim"]),
        "note": "Offline output only. No robot command was published.",
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    if flow_masks is not None:
        save_flow_masks(flow_masks, output_dir)
    print(f"Saved {args.num_samples} action chunk(s) with shape {actions_numpy.shape} to {output_dir}")


if __name__ == "__main__":
    main()
