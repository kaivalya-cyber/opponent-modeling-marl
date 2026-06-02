"""Abstract base class for all agents."""

from __future__ import annotations

import copy
import time
from abc import ABC, abstractmethod
from collections import deque
from typing import TYPE_CHECKING

import numpy as np
import torch

import config
from models.icm import ICM

if TYPE_CHECKING:
    from training.rollout_buffer import RolloutBuffer


class RunningNormalizer:
    """Running mean/std normalization with Welford's online algorithm."""

    def __init__(self, shape: tuple[int, ...], clip: float = 10.0) -> None:
        self.mean = np.zeros(shape, dtype=np.float32)
        self.var = np.ones(shape, dtype=np.float32)
        self.count = 0.0
        self.clip = clip

    def update(self, x: np.ndarray) -> None:
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        delta2 = x - self.mean
        self.var += delta * delta2

    def normalize(self, x: np.ndarray) -> np.ndarray:
        if self.count < 2:
            return x
        std = np.sqrt(self.var / self.count) + 1e-8
        return np.clip((x - self.mean) / std, -self.clip, self.clip)

    def state_dict(self) -> dict:
        return {"mean": self.mean, "var": self.var, "count": self.count, "clip": self.clip}

    def load_state_dict(self, state: dict) -> None:
        self.mean = state["mean"]
        self.var = state["var"]
        self.count = state["count"]
        self.clip = state.get("clip", self.clip)


class BaseAgent(ABC):
    """Base class every agent must implement."""

    def __init__(self) -> None:
        self.obs_normalizer: RunningNormalizer | None = None
        self.reward_normalizer: RunningNormalizer | None = None
        self.icm: ICM | None = None
        self._param_noise_std = config.PARAM_NOISE_STD
        self._original_params: list[torch.Tensor] | None = None

        # Feature 16-17: Model EMA
        self._ema_enabled = config.MODEL_EMA_ENABLED
        self._ema_tau = config.MODEL_EMA_TAU
        self._ema_policy = None
        self._ema_value = None

        # Feature 33: Episode stats tracking
        self._episode_rewards = deque(maxlen=100)
        self._episode_lengths = deque(maxlen=100)

        # Feature 35: Best episode tracking
        self._best_episode_reward = -float("inf")
        self._best_episode_length = 0

        # Feature 37: Training speed
        self._step_times = deque(maxlen=100)

        if config.OBS_NORMALIZE:
            self.obs_normalizer = RunningNormalizer((config.OBS_DIM,), config.OBS_NORM_CLIP)
        if config.REWARD_NORMALIZE:
            self.reward_normalizer = RunningNormalizer((1,), config.REWARD_NORM_CLIP)

        if config.ICM_ENABLED:
            self.icm = ICM(obs_dim=config.OBS_DIM, action_dim=config.ACTION_DIM).to(config.DEVICE)

    # ── Model EMA ──────────────────────────────────────────────────────────

    def init_ema(self) -> None:
        if not self._ema_enabled:
            return
        self._ema_policy = copy.deepcopy(self.policy)
        for p in self._ema_policy.parameters():
            p.requires_grad = False
        self._ema_value = copy.deepcopy(self.value_net)
        for p in self._ema_value.parameters():
            p.requires_grad = False

    def update_ema(self) -> None:
        if not self._ema_enabled or self._ema_policy is None:
            return
        with torch.no_grad():
            for ema_p, online_p in zip(self._ema_policy.parameters(), self.policy.parameters()):
                ema_p.data.mul_(self._ema_tau).add_(online_p.data, alpha=1.0 - self._ema_tau)
            for ema_v, online_v in zip(self._ema_value.parameters(), self.value_net.parameters()):
                ema_v.data.mul_(self._ema_tau).add_(online_v.data, alpha=1.0 - self._ema_tau)

    def select_action_ema(self, obs: np.ndarray, **kwargs) -> int:
        if not self._ema_enabled or self._ema_policy is None:
            return self.select_action_deterministic(obs, **kwargs)
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(config.DEVICE)
        with torch.no_grad():
            logits = self._ema_policy(obs_t)
        return int(logits.argmax(dim=-1).item())

    # ── Normalization ──────────────────────────────────────────────────────

    def normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        if self.obs_normalizer is not None:
            return self.obs_normalizer.normalize(obs)
        return obs

    def normalize_reward(self, reward: float) -> float:
        if self.reward_normalizer is not None:
            arr = np.array([reward], dtype=np.float32)
            self.reward_normalizer.update(arr)
            return float(self.reward_normalizer.normalize(arr)[0])
        return reward * config.REWARD_SCALE

    def update_obs_normalizer(self, obs: np.ndarray) -> None:
        if self.obs_normalizer is not None:
            self.obs_normalizer.update(obs)

    # ── Parameter noise ────────────────────────────────────────────────────

    def _apply_param_noise(self) -> None:
        if not config.PARAM_NOISE_ENABLED or self._param_noise_std <= config.PARAM_NOISE_STD_MIN:
            return
        params = []
        for module in [self.policy, self.value_net]:
            params.extend(list(module.parameters()))
        if hasattr(self, "opponent_model") and self.opponent_model is not None:
            params.extend(list(self.opponent_model.parameters()))

        if self._original_params is None:
            self._original_params = [p.data.clone() for p in params]

        for p in params:
            noise = torch.randn_like(p) * self._param_noise_std
            p.data.add_(noise)

    def _remove_param_noise(self) -> None:
        if self._original_params is None:
            return
        idx = 0
        for module in [self.policy, self.value_net]:
            for p in module.parameters():
                p.data.copy_(self._original_params[idx])
                idx += 1
        if hasattr(self, "opponent_model") and self.opponent_model is not None:
            for p in self.opponent_model.parameters():
                p.data.copy_(self._original_params[idx])
                idx += 1
        self._original_params = None

    def decay_param_noise(self) -> None:
        self._param_noise_std = max(
            self._param_noise_std * config.PARAM_NOISE_DECAY,
            config.PARAM_NOISE_STD_MIN,
        )

    # ── ICM ─────────────────────────────────────────────────────────────────

    def get_icm_intrinsic_reward(
        self, obs: np.ndarray, next_obs: np.ndarray, action: int
    ) -> float:
        if self.icm is None:
            return 0.0
        act_t = torch.tensor([action], dtype=torch.long, device=config.DEVICE)
        obs_t = torch.tensor(obs, dtype=torch.float32, device=config.DEVICE).unsqueeze(0)
        next_obs_t = torch.tensor(next_obs, dtype=torch.float32, device=config.DEVICE).unsqueeze(0)
        with torch.no_grad():
            _, _, intrinsic = self.icm(obs_t, next_obs_t, act_t)
        return config.ICM_BONUS_WEIGHT * intrinsic.item()

    def icm_update(self, batch: dict) -> tuple[float, float]:
        if self.icm is None:
            return 0.0, 0.0
        obs = batch["obs"][:, :29]
        actions = batch["actions"]
        with torch.no_grad():
            next_obs_feats = obs.clone()
        fwd_loss, inv_loss, _ = self.icm(obs, next_obs_feats, actions)
        loss = config.ICM_LOSS_WEIGHT * (config.ICM_FORWARD_LOSS_WEIGHT * fwd_loss + (1 - config.ICM_FORWARD_LOSS_WEIGHT) * inv_loss)
        return fwd_loss.item(), inv_loss.item()

    # ── Episode stats tracking (Feature 33, 35) ────────────────────────────

    def record_episode(self, reward: float, length: int) -> None:
        self._episode_rewards.append(reward)
        self._episode_lengths.append(length)
        if reward > self._best_episode_reward:
            self._best_episode_reward = reward
            self._best_episode_length = length

    def get_episode_stats(self) -> dict:
        return {
            "avg_ep_reward": np.mean(self._episode_rewards) if self._episode_rewards else 0.0,
            "avg_ep_length": np.mean(self._episode_lengths) if self._episode_lengths else 0.0,
            "best_ep_reward": self._best_episode_reward,
            "best_ep_length": self._best_episode_length,
        }

    # ── Action noise (Feature 42) ──────────────────────────────────────────

    def add_action_noise(self, action: int, action_dim: int) -> int:
        if config.ACTION_NOISE_STD <= 0:
            return action
        if np.random.random() < config.ACTION_NOISE_STD:
            return np.random.randint(0, action_dim)
        return action

    # ── Save/Load ──────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        state = {
            "policy": self.policy.state_dict(),
            "value_net": self.value_net.state_dict(),
        }
        if config.MODEL_EMA_ENABLED and self._ema_policy is not None:
            state["ema_policy"] = self._ema_policy.state_dict()
            state["ema_value"] = self._ema_value.state_dict()
        if hasattr(self, "opponent_model") and self.opponent_model is not None:
            state["opponent_model"] = self.opponent_model.state_dict()
        if self.obs_normalizer is not None:
            state["obs_normalizer"] = self.obs_normalizer.state_dict()
        if self.reward_normalizer is not None:
            state["reward_normalizer"] = self.reward_normalizer.state_dict()
        torch.save(state, path)

    def load(self, path: str) -> None:
        state = torch.load(path, map_location=config.DEVICE, weights_only=True)
        self.policy.load_state_dict(state["policy"])
        self.value_net.load_state_dict(state["value_net"])
        if config.MODEL_EMA_ENABLED and "ema_policy" in state and self._ema_policy is not None:
            self._ema_policy.load_state_dict(state["ema_policy"])
            self._ema_value.load_state_dict(state["ema_value"])
        if "opponent_model" in state and hasattr(self, "opponent_model") and self.opponent_model is not None:
            self.opponent_model.load_state_dict(state["opponent_model"])
        if "obs_normalizer" in state and self.obs_normalizer is not None:
            self.obs_normalizer.load_state_dict(state["obs_normalizer"])
        if "reward_normalizer" in state and self.reward_normalizer is not None:
            self.reward_normalizer.load_state_dict(state["reward_normalizer"])

    def _log_step_time(self) -> None:
        if config.LOG_STEPS_PER_SECOND:
            self._step_times.append(time.time())

    def get_steps_per_second(self) -> float:
        if len(self._step_times) < 2:
            return 0.0
        elapsed = self._step_times[-1] - self._step_times[0]
        return len(self._step_times) / elapsed if elapsed > 0 else 0.0

    # ── Abstract methods ───────────────────────────────────────────────────

    @abstractmethod
    def select_action(
        self, obs: np.ndarray, **kwargs
    ) -> tuple[int, torch.Tensor, torch.Tensor]:
        ...

    def select_action_deterministic(self, obs: np.ndarray, **kwargs) -> int:
        action, _, _ = self.select_action(obs, **{k: v for k, v in kwargs.items() if k != 'last_opp_action'})
        return action

    @abstractmethod
    def update(self, rollout_buffer: RolloutBuffer) -> dict:
        ...

    @abstractmethod
    def reset_episode(self) -> None:
        ...
