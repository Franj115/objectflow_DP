from __future__ import annotations

from collections.abc import Sequence
import dataclasses
import hashlib
import json
import math
from pathlib import Path
import pickle
import sys
from typing import Any

import cv2
import numpy as np
import torch
from torch import nn
import torch.nn.functional as functional
from torch.utils.data import Dataset


@dataclasses.dataclass
class NormalizationStats:
    state_mean: np.ndarray
    state_std: np.ndarray
    action_mean: np.ndarray
    action_std: np.ndarray

    def to_dict(self) -> dict[str, list[float]]:
        return {
            "state_mean": self.state_mean.tolist(),
            "state_std": self.state_std.tolist(),
            "action_mean": self.action_mean.tolist(),
            "action_std": self.action_std.tolist(),
        }

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> NormalizationStats:
        return cls(**{key: np.asarray(value, dtype=np.float32) for key, value in values.items()})


@dataclasses.dataclass
class EpisodeRecord:
    episode_id: str
    root: Path
    instruction: str
    states: np.ndarray
    actions: np.ndarray
    length: int
    flow_masks: tuple[Path, ...] = ()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_huggingface_pretrained(factory: Any, model_name: str) -> Any:
    """Use an existing HF cache offline, falling back to the normal download path."""
    try:
        return factory.from_pretrained(model_name, local_files_only=True)
    except OSError:
        return factory.from_pretrained(model_name)


def parse_episode_spec(spec: str) -> list[str]:
    """Parse values such as ``000-013,020`` into zero-padded episode IDs."""
    result: list[str] = []
    for raw_item in spec.split(","):
        item = raw_item.strip()
        if not item:
            continue
        if "-" in item:
            start_text, end_text = item.split("-", maxsplit=1)
            start, end = int(start_text), int(end_text)
            if end < start:
                raise ValueError(f"Invalid episode range: {item}")
            result.extend(f"{value:03d}" for value in range(start, end + 1))
        else:
            result.append(f"{int(item):03d}")
    if not result:
        raise ValueError("Episode selection is empty.")
    if len(result) != len(set(result)):
        raise ValueError(f"Episode selection contains duplicates: {spec}")
    return result


def load_instruction_map(data_root: Path) -> dict[str, str]:
    path = data_root / "language_instruction.json"
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    result: dict[str, str] = {}
    for key, raw_value in raw.items():
        value = raw_value.get("language_instruction", "") if isinstance(raw_value, dict) else raw_value
        result[f"{int(key):03d}"] = str(value)
    return result


def _video_frame_count(path: Path) -> int:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    return count


class DualViewActionDataset(Dataset):
    """Aligned dual-view RGB, state history, action chunks, and optional GT flow masks."""

    def __init__(
        self,
        data_root: str | Path,
        episode_ids: Sequence[str],
        *,
        obs_horizon: int,
        action_horizon: int,
        image_size: int,
        state_key: str = "pose",
        action_key: str = "action",
        flow_root: str | Path | None = None,
        require_flow_masks: bool = False,
        strict_alignment: bool = True,
    ) -> None:
        self.data_root = Path(data_root)
        self.episode_ids = [f"{int(value):03d}" for value in episode_ids]
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon
        self.image_size = image_size
        self.state_key = state_key
        self.action_key = action_key
        self.flow_root = Path(flow_root) if flow_root is not None else None
        self.require_flow_masks = require_flow_masks
        self.strict_alignment = strict_alignment
        self._captures: dict[Path, cv2.VideoCapture] = {}

        instructions = load_instruction_map(self.data_root)
        self.episodes: list[EpisodeRecord] = []
        self.index: list[tuple[int, int]] = []
        for episode_id in self.episode_ids:
            root = self.data_root / f"epi_{episode_id}"
            trajectory_path = root / "trajectory.pkl"
            scene_path = root / "scene_camera.mp4"
            wrist_path = root / "wrist_camera.mp4"
            for required in (trajectory_path, scene_path, wrist_path):
                if not required.exists():
                    raise FileNotFoundError(required)

            with trajectory_path.open("rb") as file:
                trajectory = pickle.load(file)
            if state_key not in trajectory or action_key not in trajectory:
                raise KeyError(
                    f"{trajectory_path} needs keys {state_key!r} and {action_key!r}; found {sorted(trajectory)}"
                )
            states = np.asarray(trajectory[state_key], dtype=np.float32)
            actions = np.asarray(trajectory[action_key], dtype=np.float32)
            if states.ndim != 2 or actions.ndim != 2:
                raise ValueError(f"Expected 2-D state/action arrays in {trajectory_path}.")

            lengths = {
                "state": len(states),
                "action": len(actions),
                "scene": _video_frame_count(scene_path),
                "wrist": _video_frame_count(wrist_path),
            }
            flow_masks: tuple[Path, ...] = ()
            if self.flow_root is not None:
                mask_dir = self.flow_root / f"episode_{episode_id}" / "masks"
                if mask_dir.exists():
                    flow_masks = tuple(
                        sorted(path for path in mask_dir.iterdir() if path.suffix.lower() in {".png", ".jpg", ".jpeg"})
                    )
                    lengths["flow"] = len(flow_masks)
                elif require_flow_masks:
                    raise FileNotFoundError(mask_dir)

            if strict_alignment and len(set(lengths.values())) != 1:
                raise ValueError(f"Episode {episode_id} is not frame-aligned: {lengths}")
            length = min(lengths.values())
            if length < obs_horizon + action_horizon - 1:
                raise ValueError(f"Episode {episode_id} is too short: {length} frames")

            record = EpisodeRecord(
                episode_id=episode_id,
                root=root,
                instruction=instructions.get(episode_id, ""),
                states=states[:length],
                actions=actions[:length],
                length=length,
                flow_masks=flow_masks[:length],
            )
            episode_index = len(self.episodes)
            self.episodes.append(record)
            first_t = obs_horizon - 1
            last_t = length - action_horizon
            self.index.extend((episode_index, t) for t in range(first_t, last_t + 1))

    @property
    def state_dim(self) -> int:
        return int(self.episodes[0].states.shape[-1])

    @property
    def action_dim(self) -> int:
        return int(self.episodes[0].actions.shape[-1])

    def __len__(self) -> int:
        return len(self.index)

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_captures"] = {}
        return state

    def _read_frame(self, path: Path, frame_index: int) -> torch.Tensor:
        capture = self._captures.get(path)
        if capture is None:
            capture = cv2.VideoCapture(str(path))
            if not capture.isOpened():
                raise RuntimeError(f"Cannot open video: {path}")
            self._captures[path] = capture
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError(f"Cannot read frame {frame_index} from {path}")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        return torch.from_numpy(np.ascontiguousarray(frame)).permute(2, 0, 1).float().div_(255.0)

    def _read_mask(self, path: Path) -> torch.Tensor:
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"Cannot read mask: {path}")
        mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        return torch.from_numpy((mask > 127).astype(np.float32)).unsqueeze(0)

    def __getitem__(self, item: int) -> dict[str, Any]:
        episode_index, t = self.index[item]
        episode = self.episodes[episode_index]
        observation_indices = range(t - self.obs_horizon + 1, t + 1)
        agent_rgb = torch.stack(
            [self._read_frame(episode.root / "scene_camera.mp4", frame_index) for frame_index in observation_indices]
        )
        wrist_rgb = torch.stack(
            [self._read_frame(episode.root / "wrist_camera.mp4", frame_index) for frame_index in observation_indices]
        )
        sample: dict[str, Any] = {
            "agent_rgb": agent_rgb,
            "wrist_rgb": wrist_rgb,
            "robot_state": torch.from_numpy(episode.states[t - self.obs_horizon + 1 : t + 1].copy()),
            "action": torch.from_numpy(episode.actions[t : t + self.action_horizon].copy()),
            "language": episode.instruction,
            "episode_id": episode.episode_id,
            "frame_index": t,
        }
        if episode.flow_masks:
            sample["flow_mask"] = torch.stack(
                [self._read_mask(path) for path in episode.flow_masks[t : t + self.action_horizon]]
            )
        return sample


def compute_normalization_stats(dataset: DualViewActionDataset, eps: float = 1e-6) -> NormalizationStats:
    states = np.concatenate([episode.states for episode in dataset.episodes], axis=0)
    actions = np.concatenate([episode.actions for episode in dataset.episodes], axis=0)
    return NormalizationStats(
        state_mean=states.mean(axis=0).astype(np.float32),
        state_std=np.maximum(states.std(axis=0), eps).astype(np.float32),
        action_mean=actions.mean(axis=0).astype(np.float32),
        action_std=np.maximum(actions.std(axis=0), eps).astype(np.float32),
    )


def normalize_tensor(value: torch.Tensor, mean: np.ndarray, std: np.ndarray) -> torch.Tensor:
    mean_tensor = torch.as_tensor(mean, device=value.device, dtype=value.dtype)
    std_tensor = torch.as_tensor(std, device=value.device, dtype=value.dtype)
    return (value - mean_tensor) / std_tensor


def denormalize_tensor(value: torch.Tensor, mean: np.ndarray, std: np.ndarray) -> torch.Tensor:
    mean_tensor = torch.as_tensor(mean, device=value.device, dtype=value.dtype)
    std_tensor = torch.as_tensor(std, device=value.device, dtype=value.dtype)
    return value * std_tensor + mean_tensor


class InstructionTokenizer:
    def __init__(self, encoder: str, model_name: str, max_length: int, hash_vocab_size: int = 8192) -> None:
        self.encoder = encoder
        self.max_length = max_length
        self.hash_vocab_size = hash_vocab_size
        if encoder == "bert":
            from transformers import AutoTokenizer

            self.tokenizer = load_huggingface_pretrained(AutoTokenizer, model_name)
        elif encoder == "hash":
            self.tokenizer = None
        else:
            raise ValueError(f"Unknown language encoder: {encoder}")

    def __call__(self, texts: Sequence[str], device: torch.device) -> dict[str, torch.Tensor]:
        if self.encoder == "bert":
            encoded = self.tokenizer(
                list(texts),
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            return {key: value.to(device) for key, value in encoded.items()}

        rows = []
        for text in texts:
            ids = []
            for token in text.lower().replace(".", " ").replace(",", " ").split()[: self.max_length]:
                digest = hashlib.sha1(token.encode("utf-8")).digest()
                ids.append(1 + int.from_bytes(digest[:4], "little") % (self.hash_vocab_size - 1))
            ids.extend([0] * (self.max_length - len(ids)))
            rows.append(ids)
        input_ids = torch.tensor(rows, dtype=torch.long, device=device)
        return {"input_ids": input_ids, "attention_mask": input_ids.ne(0).long()}


class SmallVisionEncoder(nn.Module):
    def __init__(self, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.projection = nn.Linear(128, output_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.projection(self.net(images).flatten(1))


class FrozenDinoEncoder(nn.Module):
    def __init__(self, output_dim: int, model_name: str) -> None:
        super().__init__()
        self.backbone = torch.hub.load("facebookresearch/dinov2", model_name)
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(requires_grad=False)
        dino_dim = int(getattr(self.backbone, "embed_dim", 384))
        self.projection = nn.Linear(dino_dim, output_dim)
        self.register_buffer("image_mean", torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1))
        self.register_buffer("image_std", torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        self.backbone.eval()
        with torch.no_grad():
            normalized = (images - self.image_mean) / self.image_std
            features = self.backbone.forward_features(normalized)
            if isinstance(features, dict) and "x_norm_clstoken" in features:
                features = features["x_norm_clstoken"]
            elif isinstance(features, dict) and "x_norm_patchtokens" in features:
                features = features["x_norm_patchtokens"].mean(dim=1)
        return self.projection(features)


class TextEncoder(nn.Module):
    def __init__(self, output_dim: int, encoder: str, model_name: str, hash_vocab_size: int = 8192) -> None:
        super().__init__()
        self.encoder = encoder
        if encoder == "bert":
            from transformers import AutoModel

            self.backbone = load_huggingface_pretrained(AutoModel, model_name)
            for parameter in self.backbone.parameters():
                parameter.requires_grad_(requires_grad=False)
            self.embedding = None
            self.projection = nn.Linear(int(self.backbone.config.hidden_size), output_dim)
        elif encoder == "hash":
            self.backbone = None
            self.embedding = nn.Embedding(hash_vocab_size, output_dim, padding_idx=0)
            self.projection = nn.Sequential(nn.Linear(output_dim, output_dim), nn.SiLU())
        else:
            raise ValueError(f"Unknown language encoder: {encoder}")

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.encoder == "bert":
            self.backbone.eval()
            with torch.no_grad():
                output = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
            features = output.last_hidden_state[:, 0]
        else:
            tokens = self.embedding(input_ids)
            valid = attention_mask.unsqueeze(-1).to(tokens.dtype)
            features = (tokens * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
        return self.projection(features)


class ObservationEncoder(nn.Module):
    def __init__(
        self,
        *,
        obs_horizon: int,
        state_dim: int,
        condition_dim: int,
        vision_encoder: str,
        dinov2_model: str,
        language_encoder: str,
        bert_model: str,
        token_layers: int,
    ) -> None:
        super().__init__()
        self.obs_horizon = obs_horizon
        if vision_encoder == "dinov2":
            self.vision = FrozenDinoEncoder(condition_dim, dinov2_model)
        elif vision_encoder == "small_cnn":
            self.vision = SmallVisionEncoder(condition_dim)
        else:
            raise ValueError(f"Unknown vision encoder: {vision_encoder}")
        self.text = TextEncoder(condition_dim, language_encoder, bert_model)
        self.camera_embedding = nn.Parameter(torch.randn(2, condition_dim) * 0.02)
        self.time_embedding = nn.Parameter(torch.randn(obs_horizon, condition_dim) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=condition_dim,
            nhead=8,
            dim_feedforward=condition_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.visual_transformer = nn.TransformerEncoder(layer, num_layers=token_layers)
        self.state_encoder = nn.Sequential(
            nn.Linear(obs_horizon * state_dim, condition_dim),
            nn.SiLU(),
            nn.Linear(condition_dim, condition_dim),
        )
        self.fusion = nn.Sequential(
            nn.Linear(condition_dim * 3, condition_dim * 2),
            nn.SiLU(),
            nn.Linear(condition_dim * 2, condition_dim),
        )

    def forward(
        self,
        agent_rgb: torch.Tensor,
        wrist_rgb: torch.Tensor,
        robot_state: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, obs_horizon = agent_rgb.shape[:2]
        if obs_horizon != self.obs_horizon:
            raise ValueError(f"Expected {self.obs_horizon} observation frames, got {obs_horizon}")
        views = torch.stack((agent_rgb, wrist_rgb), dim=2)
        frames = views.reshape(-1, *views.shape[-3:])
        visual = self.vision(frames).reshape(batch_size, obs_horizon, 2, -1)
        visual = visual + self.time_embedding[None, :, None, :] + self.camera_embedding[None, None, :, :]
        visual = self.visual_transformer(visual.reshape(batch_size, obs_horizon * 2, -1)).mean(dim=1)
        state = self.state_encoder(robot_state.flatten(1))
        text = self.text(input_ids, attention_mask)
        return self.fusion(torch.cat((visual, state, text), dim=-1))


class FlowEncoder(nn.Module):
    """Encode every future mask separately before temporal attention."""

    def __init__(self, flow_horizon: int, condition_dim: int, token_layers: int = 2) -> None:
        super().__init__()
        self.flow_horizon = flow_horizon
        self.frame_encoder = nn.Sequential(
            nn.Conv2d(1, 32, 5, stride=2, padding=2),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, condition_dim, 3, stride=2, padding=1),
            nn.GroupNorm(8, condition_dim),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.time_embedding = nn.Parameter(torch.randn(flow_horizon, condition_dim) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=condition_dim,
            nhead=8,
            dim_feedforward=condition_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_transformer = nn.TransformerEncoder(layer, num_layers=token_layers)

    def forward(self, masks: torch.Tensor) -> torch.Tensor:
        batch_size, horizon = masks.shape[:2]
        if horizon != self.flow_horizon:
            raise ValueError(f"Expected {self.flow_horizon} flow frames, got {horizon}")
        features = self.frame_encoder(masks.flatten(0, 1)).flatten(1).reshape(batch_size, horizon, -1)
        features = self.temporal_transformer(features + self.time_embedding[None])
        return features.mean(dim=1)


def sinusoidal_embedding(timesteps: torch.Tensor, dimension: int) -> torch.Tensor:
    half = dimension // 2
    scale = math.log(10_000) / max(half - 1, 1)
    frequencies = torch.exp(torch.arange(half, device=timesteps.device, dtype=torch.float32) * -scale)
    angles = timesteps.float()[:, None] * frequencies[None]
    embedding = torch.cat((angles.sin(), angles.cos()), dim=-1)
    if dimension % 2:
        embedding = functional.pad(embedding, (0, 1))
    return embedding


class ConditionalResidualBlock1D(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, condition_dim: int) -> None:
        super().__init__()
        self.block1 = nn.Sequential(
            nn.GroupNorm(8, input_channels),
            nn.SiLU(),
            nn.Conv1d(input_channels, output_channels, 3, padding=1),
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(8, output_channels),
            nn.SiLU(),
            nn.Conv1d(output_channels, output_channels, 3, padding=1),
        )
        self.film = nn.Linear(condition_dim, output_channels * 2)
        self.skip = (
            nn.Conv1d(input_channels, output_channels, 1) if input_channels != output_channels else nn.Identity()
        )

    def forward(self, sample: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        hidden = self.block1(sample)
        scale, bias = self.film(condition).chunk(2, dim=-1)
        hidden = hidden * (1.0 + scale[..., None]) + bias[..., None]
        hidden = self.block2(hidden)
        return hidden + self.skip(sample)


class ConditionalActionUNet1D(nn.Module):
    def __init__(self, action_dim: int, condition_dim: int, base_channels: int) -> None:
        super().__init__()
        if base_channels % 8:
            raise ValueError("base_channels must be divisible by 8")
        self.condition_dim = condition_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(condition_dim, condition_dim * 2),
            nn.SiLU(),
            nn.Linear(condition_dim * 2, condition_dim),
        )
        self.condition_mlp = nn.Sequential(nn.Linear(condition_dim, condition_dim), nn.SiLU())
        self.input_projection = nn.Conv1d(action_dim, base_channels, 3, padding=1)
        self.down_block1 = ConditionalResidualBlock1D(base_channels, base_channels, condition_dim)
        self.downsample1 = nn.Conv1d(base_channels, base_channels * 2, 4, stride=2, padding=1)
        self.down_block2 = ConditionalResidualBlock1D(base_channels * 2, base_channels * 2, condition_dim)
        self.downsample2 = nn.Conv1d(base_channels * 2, base_channels * 4, 4, stride=2, padding=1)
        self.middle1 = ConditionalResidualBlock1D(base_channels * 4, base_channels * 4, condition_dim)
        self.middle2 = ConditionalResidualBlock1D(base_channels * 4, base_channels * 4, condition_dim)
        self.upsample1 = nn.ConvTranspose1d(base_channels * 4, base_channels * 2, 4, stride=2, padding=1)
        self.up_block1 = ConditionalResidualBlock1D(base_channels * 4, base_channels * 2, condition_dim)
        self.upsample2 = nn.ConvTranspose1d(base_channels * 2, base_channels, 4, stride=2, padding=1)
        self.up_block2 = ConditionalResidualBlock1D(base_channels * 2, base_channels, condition_dim)
        self.output_projection = nn.Sequential(
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.Conv1d(base_channels, action_dim, 3, padding=1),
        )

    def forward(self, noisy_actions: torch.Tensor, timesteps: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        if noisy_actions.shape[1] % 4:
            raise ValueError("Action horizon must be divisible by 4 for this denoiser.")
        conditioning = self.time_mlp(sinusoidal_embedding(timesteps, self.condition_dim))
        conditioning = conditioning + self.condition_mlp(condition)
        sample = noisy_actions.transpose(1, 2)
        level1 = self.down_block1(self.input_projection(sample), conditioning)
        level2 = self.down_block2(self.downsample1(level1), conditioning)
        middle = self.middle2(self.middle1(self.downsample2(level2), conditioning), conditioning)
        sample = self.upsample1(middle)
        sample = self.up_block1(torch.cat((sample, level2), dim=1), conditioning)
        sample = self.upsample2(sample)
        sample = self.up_block2(torch.cat((sample, level1), dim=1), conditioning)
        return self.output_projection(sample).transpose(1, 2)


class DiffusionPolicy(nn.Module):
    def __init__(
        self,
        *,
        obs_horizon: int,
        state_dim: int,
        action_dim: int,
        action_horizon: int,
        condition_dim: int,
        base_channels: int,
        vision_encoder: str,
        dinov2_model: str,
        language_encoder: str,
        bert_model: str,
        observation_token_layers: int,
        use_flow: bool,
        flow_token_layers: int,
    ) -> None:
        super().__init__()
        self.use_flow = use_flow
        self.observation_encoder = ObservationEncoder(
            obs_horizon=obs_horizon,
            state_dim=state_dim,
            condition_dim=condition_dim,
            vision_encoder=vision_encoder,
            dinov2_model=dinov2_model,
            language_encoder=language_encoder,
            bert_model=bert_model,
            token_layers=observation_token_layers,
        )
        if use_flow:
            self.flow_encoder = FlowEncoder(action_horizon, condition_dim, flow_token_layers)
            self.flow_fusion = nn.Sequential(
                nn.Linear(condition_dim * 2, condition_dim * 2),
                nn.SiLU(),
                nn.Linear(condition_dim * 2, condition_dim),
            )
        else:
            self.flow_encoder = None
            self.flow_fusion = None
        self.denoiser = ConditionalActionUNet1D(action_dim, condition_dim, base_channels)

    def encode_condition(
        self,
        agent_rgb: torch.Tensor,
        wrist_rgb: torch.Tensor,
        robot_state: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        flow_masks: torch.Tensor | None = None,
    ) -> torch.Tensor:
        observation = self.observation_encoder(agent_rgb, wrist_rgb, robot_state, input_ids, attention_mask)
        if not self.use_flow:
            return observation
        if flow_masks is None:
            raise ValueError("Flow-conditioned policy requires flow_masks.")
        flow = self.flow_encoder(flow_masks)
        return self.flow_fusion(torch.cat((observation, flow), dim=-1))

    def forward(
        self,
        noisy_actions: torch.Tensor,
        timesteps: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        return self.denoiser(noisy_actions, timesteps, condition)


class DDPMSchedule(nn.Module):
    def __init__(self, num_steps: int, beta_start: float, beta_end: float) -> None:
        super().__init__()
        if num_steps < 2:
            raise ValueError("num_steps must be at least 2")
        betas = torch.linspace(beta_start, beta_end, num_steps, dtype=torch.float32)
        alphas = 1.0 - betas
        alpha_cumprod = torch.cumprod(alphas, dim=0)
        alpha_cumprod_previous = functional.pad(alpha_cumprod[:-1], (1, 0), value=1.0)
        self.num_steps = num_steps
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_cumprod", alpha_cumprod)
        self.register_buffer("alpha_cumprod_previous", alpha_cumprod_previous)

    def add_noise(self, clean: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        shape = (len(timesteps),) + (1,) * (clean.ndim - 1)
        alpha = self.alpha_cumprod[timesteps].reshape(shape)
        return alpha.sqrt() * clean + (1.0 - alpha).sqrt() * noise

    @torch.no_grad()
    def sample(
        self,
        model: DiffusionPolicy,
        condition: torch.Tensor,
        *,
        action_horizon: int,
        action_dim: int,
        clip_denoised: float | None = 5.0,
    ) -> torch.Tensor:
        sample = torch.randn(len(condition), action_horizon, action_dim, device=condition.device)
        for step in reversed(range(self.num_steps)):
            timesteps = torch.full((len(condition),), step, device=condition.device, dtype=torch.long)
            predicted_noise = model(sample, timesteps, condition)
            alpha_cumprod = self.alpha_cumprod[step]
            predicted_clean = (sample - (1.0 - alpha_cumprod).sqrt() * predicted_noise) / alpha_cumprod.sqrt()
            if clip_denoised is not None:
                predicted_clean = predicted_clean.clamp(-clip_denoised, clip_denoised)
            previous_alpha = self.alpha_cumprod_previous[step]
            beta = self.betas[step]
            alpha = self.alphas[step]
            denominator = 1.0 - alpha_cumprod
            mean = (
                beta * previous_alpha.sqrt() / denominator * predicted_clean
                + (1.0 - previous_alpha) * alpha.sqrt() / denominator * sample
            )
            if step > 0:
                variance = beta * (1.0 - previous_alpha) / denominator
                sample = mean + variance.clamp_min(1e-20).sqrt() * torch.randn_like(sample)
            else:
                sample = mean
        return sample


def policy_from_config(config: dict[str, Any], *, use_flow: bool) -> DiffusionPolicy:
    return DiffusionPolicy(
        obs_horizon=int(config["obs_horizon"]),
        state_dim=int(config["state_dim"]),
        action_dim=int(config["action_dim"]),
        action_horizon=int(config["action_horizon"]),
        condition_dim=int(config["condition_dim"]),
        base_channels=int(config["base_channels"]),
        vision_encoder=str(config["vision_encoder"]),
        dinov2_model=str(config["dinov2_model"]),
        language_encoder=str(config["language_encoder"]),
        bert_model=str(config["bert_model"]),
        observation_token_layers=int(config["observation_token_layers"]),
        use_flow=use_flow,
        flow_token_layers=int(config.get("flow_token_layers", 2)),
    )


def trainable_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    return {name: parameter.detach().cpu() for name, parameter in module.named_parameters() if parameter.requires_grad}


def load_trainable_state_dict(module: nn.Module, state_dict: dict[str, torch.Tensor]) -> None:
    missing, unexpected = module.load_state_dict(state_dict, strict=False)
    trainable_names = {name for name, parameter in module.named_parameters() if parameter.requires_grad}
    missing_trainable = sorted(trainable_names.intersection(missing))
    if missing_trainable or unexpected:
        raise RuntimeError(f"Checkpoint mismatch: missing trainable={missing_trainable}, unexpected={unexpected}")


class FrozenObjectFlowAdapter:
    """Load the existing ObjectFlowPredictor without embedding it in policy checkpoints."""

    def __init__(
        self,
        checkpoint: str | Path,
        project_root: str | Path,
        device: torch.device,
        vocab_path: str | Path | None = None,
    ) -> None:
        self.checkpoint_path = Path(checkpoint)
        project_root = Path(project_root).resolve()
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        from object_flow.model import ObjectFlowPredictor

        saved = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        self.args = dict(saved["args"])
        self.device = device
        self.language_encoder = self.args.get("language_encoder", "simple")
        self.max_text_len = int(self.args.get("max_text_len", 32))
        self.image_size = int(self.args.get("image_size", 128))
        self.model = ObjectFlowPredictor(
            vocab_size=int(saved.get("vocab_size", 2)),
            obs_horizon=int(self.args["obs_horizon"]),
            flow_horizon=int(self.args["flow_horizon"]),
            image_size=self.image_size,
            hidden_dim=int(self.args["hidden_dim"]),
            num_layers=int(self.args["num_layers"]),
            max_text_len=self.max_text_len,
            image_encoder=self.args["image_encoder"],
            language_encoder=self.language_encoder,
            dinov2_model=self.args.get("dinov2_model", "dinov2_vits14"),
            bert_model=self.args.get("bert_model", "bert-base-uncased"),
        ).to(device)
        self.model.load_state_dict(saved["model"])
        self.model.requires_grad_(requires_grad=False).eval()

        if self.language_encoder == "bert":
            from transformers import AutoTokenizer

            self.tokenizer = load_huggingface_pretrained(
                AutoTokenizer, self.args.get("bert_model", "bert-base-uncased")
            )
            self.vocab = None
        else:
            path = Path(vocab_path) if vocab_path is not None else self.checkpoint_path.parent / "vocab.json"
            self.vocab = json.loads(path.read_text(encoding="utf-8"))
            self.tokenizer = None

    def _simple_tokens(self, instructions: Sequence[str]) -> torch.Tensor:
        rows = []
        for text in instructions:
            tokens = text.lower().replace(".", " ").replace(",", " ").split()[: self.max_text_len]
            ids = [self.vocab.get(token, 1) for token in tokens]
            ids.extend([0] * (self.max_text_len - len(ids)))
            rows.append(ids)
        return torch.tensor(rows, device=self.device, dtype=torch.long)

    @torch.no_grad()
    def predict(self, agent_rgb: torch.Tensor, instructions: Sequence[str]) -> torch.Tensor:
        images = agent_rgb.to(self.device)
        if images.shape[-1] != self.image_size or images.shape[-2] != self.image_size:
            images = functional.interpolate(
                images.flatten(0, 1),
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            ).reshape(images.shape[0], images.shape[1], 3, self.image_size, self.image_size)
        if self.language_encoder == "bert":
            encoded = self.tokenizer(
                list(instructions),
                padding="max_length",
                truncation=True,
                max_length=self.max_text_len,
                return_tensors="pt",
            )
            logits = self.model(
                images,
                bert_input_ids=encoded["input_ids"].to(self.device),
                bert_attention_mask=encoded["attention_mask"].to(self.device),
            )
        else:
            logits = self.model(images, text=self._simple_tokens(instructions))
        return logits.sigmoid()


def load_single_observation(
    episode_dir: str | Path,
    *,
    frame_index: int,
    obs_horizon: int,
    image_size: int,
    state_key: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    episode_dir = Path(episode_dir)
    if frame_index < obs_horizon - 1:
        raise ValueError(f"frame_index must be at least {obs_horizon - 1}")
    with (episode_dir / "trajectory.pkl").open("rb") as file:
        trajectory = pickle.load(file)
    states = np.asarray(trajectory[state_key], dtype=np.float32)
    if frame_index >= len(states):
        raise IndexError(f"frame_index {frame_index} exceeds trajectory length {len(states)}")

    def read_history(path: Path) -> torch.Tensor:
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open video: {path}")
        frames = []
        for index in range(frame_index - obs_horizon + 1, frame_index + 1):
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"Cannot read frame {index} from {path}")
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (image_size, image_size), interpolation=cv2.INTER_AREA)
            frames.append(torch.from_numpy(np.ascontiguousarray(frame)).permute(2, 0, 1).float() / 255.0)
        capture.release()
        return torch.stack(frames)

    agent = read_history(episode_dir / "scene_camera.mp4")
    wrist = read_history(episode_dir / "wrist_camera.mp4")
    state = torch.from_numpy(states[frame_index - obs_horizon + 1 : frame_index + 1].copy())
    return agent, wrist, state


def instruction_for_episode(episode_dir: str | Path) -> str:
    episode_dir = Path(episode_dir)
    episode_id = episode_dir.name.rsplit("_", maxsplit=1)[-1]
    return load_instruction_map(episode_dir.parent).get(f"{int(episode_id):03d}", "")
