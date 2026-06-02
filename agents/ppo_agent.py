"""Standard PPO agent — baseline without opponent modeling."""

from __future__ import annotations

from collections import deque

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

import config
from agents.base_agent import BaseAgent
from models.policy_net import PolicyNet
from models.value_net import ValueNet
from training.rollout_buffer import RolloutBuffer


class PPOAgent(BaseAgent):
    """Proximal Policy Optimization agent with optional enhancements."""

    def __init__(self, obs_dim: int = 29, action_dim: int = 5) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self._frame_stack_size = config.FRAME_STACK
        self._frame_stack: deque | None = None
        self._effective_obs_dim = obs_dim * self._frame_stack_size

        self.policy = PolicyNet(input_dim=self._effective_obs_dim, action_dim=action_dim).to(config.DEVICE)
        self.value_net = ValueNet(input_dim=self._effective_obs_dim).to(config.DEVICE)

        # Feature 4-6: Separate optimizers
        self._use_separate_opt = config.USE_SEPARATE_OPTIMIZERS
        if self._use_separate_opt:
            self.policy_optimizer = torch.optim.Adam(
                self.policy.parameters(), lr=config.POLICY_LR,
                weight_decay=config.POLICY_WEIGHT_DECAY,
            )
            self.value_optimizer = torch.optim.Adam(
                self.value_net.parameters(), lr=config.VALUE_LR,
                weight_decay=config.VALUE_WEIGHT_DECAY,
            )
            self.optimizer = self.policy_optimizer
        else:
            self.optimizer = torch.optim.Adam(
                list(self.policy.parameters()) + list(self.value_net.parameters()),
                lr=config.LEARNING_RATE,
                weight_decay=config.POLICY_WEIGHT_DECAY,
            )

        # Feature 16-17: Model EMA
        self.init_ema()

        # Feature 26-27: Scheduled entropy and clip epsilon
        self._current_entropy_coeff = config.ENTROPY_COEFF
        self._current_clip_epsilon = config.CLIP_EPSILON
        self._total_updates_made = 0

        # Learning rate scheduler
        self.lr_scheduler = self._build_lr_scheduler()
        self._lr_warmup_steps = config.LR_WARMUP_STEPS
        self._total_updates = 0

        self._grad_accum_steps = config.GRAD_ACCUMULATION_STEPS

    def _build_lr_scheduler(self):
        base_lr = config.LEARNING_RATE
        total_steps = config.TOTAL_EPISODES * config.N_EPOCHS * (config.ROLLOUT_STEPS // config.MINIBATCH_SIZE)
        min_lr = base_lr * config.LR_MIN_RATIO

        if config.LR_SCHEDULER_TYPE == "none":
            return None
        elif config.LR_SCHEDULER_TYPE == "cosine":
            if config.LR_WARMUP_STEPS > 0:
                warmup = LinearLR(self.optimizer, start_factor=0.01, end_factor=1.0,
                                  total_iters=config.LR_WARMUP_STEPS)
                cosine = CosineAnnealingLR(self.optimizer, T_max=max(1, total_steps - config.LR_WARMUP_STEPS),
                                           eta_min=min_lr)
                return SequentialLR(self.optimizer, schedulers=[warmup, cosine],
                                    milestones=[config.LR_WARMUP_STEPS])
            else:
                return CosineAnnealingLR(self.optimizer, T_max=max(1, total_steps), eta_min=min_lr)
        elif config.LR_SCHEDULER_TYPE == "linear":
            if config.LR_WARMUP_STEPS > 0:
                warmup = LinearLR(self.optimizer, start_factor=0.01, end_factor=1.0,
                                  total_iters=config.LR_WARMUP_STEPS)
                linear = LinearLR(self.optimizer, start_factor=1.0, end_factor=config.LR_MIN_RATIO,
                                  total_iters=max(1, total_steps - config.LR_WARMUP_STEPS))
                return SequentialLR(self.optimizer, schedulers=[warmup, linear],
                                    milestones=[config.LR_WARMUP_STEPS])
            else:
                return LinearLR(self.optimizer, start_factor=1.0, end_factor=config.LR_MIN_RATIO,
                                total_iters=max(1, total_steps))
        return None

    def _build_frame_stack_obs(self, obs: np.ndarray) -> np.ndarray:
        if self._frame_stack_size <= 1:
            return obs
        obs_t = torch.tensor(obs, dtype=torch.float32)
        self._frame_stack.append(obs_t)
        stacked = torch.cat(list(self._frame_stack), dim=-1).numpy()
        assert stacked.shape == (self._effective_obs_dim,), (
            f"Stacked obs shape mismatch: {stacked.shape}"
        )
        return stacked

    def select_action(
        self, obs: np.ndarray, **kwargs
    ) -> tuple[int, torch.Tensor, torch.Tensor]:
        obs = self.normalize_obs(obs)
        self.update_obs_normalizer(obs)
        obs_stacked = self._build_frame_stack_obs(obs)

        obs_t = torch.tensor(obs_stacked, dtype=torch.float32).unsqueeze(0).to(config.DEVICE)
        with torch.no_grad():
            logits = self.policy(obs_t, temperature=config.SOFTMAX_TEMPERATURE)
            value = self.value_net(obs_t)

        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        # Feature 42: Action noise
        action_int = action.item()
        action_int = self.add_action_noise(action_int, self.action_dim)

        return action_int, log_prob.squeeze(0).cpu(), value.squeeze(0).cpu()

    def select_action_deterministic(self, obs: np.ndarray, **kwargs) -> int:
        obs = self.normalize_obs(obs)
        obs_stacked = self._build_frame_stack_obs(obs)
        obs_t = torch.tensor(obs_stacked, dtype=torch.float32).unsqueeze(0).to(config.DEVICE)
        with torch.no_grad():
            logits = self.policy(obs_t)
        return int(logits.argmax(dim=-1).item())

    def update(self, rollout_buffer: RolloutBuffer) -> dict:
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_approx_kl = 0.0
        total_explained_var = 0.0
        n_updates = 0

        # Feature 26: Scheduled entropy
        if config.ENTROPY_SCHEDULE_STEPS > 0 and self._total_updates_made < config.ENTROPY_SCHEDULE_STEPS:
            progress = self._total_updates_made / config.ENTROPY_SCHEDULE_STEPS
            self._current_entropy_coeff = config.ENTROPY_COEFF * (1.0 - progress) + config.ENTROPY_TARGET * progress

        # Feature 27: Scheduled clip epsilon
        if config.CLIP_EPSILON_SCHEDULE_STEPS > 0:
            progress = min(1.0, self._total_updates_made / config.CLIP_EPSILON_SCHEDULE_STEPS)
            self._current_clip_epsilon = config.CLIP_EPSILON * (1.0 - progress) + 0.05 * progress

        opt = self.optimizer
        opt.zero_grad()

        for epoch in range(config.N_EPOCHS):
            for batch_idx, batch in enumerate(rollout_buffer.get_batches(config.MINIBATCH_SIZE)):
                obs = batch["obs"]
                actions = batch["actions"]
                old_log_probs = batch["log_probs"]
                advantages = batch["advantages"]
                returns = batch["returns"]

                if self.obs_dim < obs.shape[-1]:
                    obs = obs[:, :self.obs_dim]

                adv = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                logits = self.policy(obs, temperature=config.SOFTMAX_TEMPERATURE)
                values = self.value_net(obs).squeeze(-1)

                # Feature 9: Value residual clip
                if config.VALUE_RESIDUAL_CLIP > 0:
                    values = returns + torch.clamp(values - returns, -config.VALUE_RESIDUAL_CLIP, config.VALUE_RESIDUAL_CLIP)

                dist = Categorical(logits=logits)
                new_log_probs = dist.log_prob(actions)
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_log_probs - old_log_probs)
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1.0 - self._current_clip_epsilon, 1.0 + self._current_clip_epsilon) * adv
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(values, returns)

                # Feature 24-25: Approx KL and explained variance
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - (new_log_probs - old_log_probs)).mean().item()
                    var_explained = 1.0 - (F.mse_loss(values, returns).item() / (returns.var() + 1e-8).item())
                    total_approx_kl += approx_kl
                    total_explained_var += var_explained

                loss = (
                    policy_loss
                    + config.VALUE_LOSS_COEFF * value_loss
                    - self._current_entropy_coeff * entropy
                )

                loss = loss / self._grad_accum_steps
                loss.backward()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                n_updates += 1

                if (batch_idx + 1) % self._grad_accum_steps == 0:
                    self._apply_grad_clipping()
                    opt.step()
                    opt.zero_grad()

                    # Feature 16-17: Model EMA update
                    self.update_ema()

                    if self.lr_scheduler is not None:
                        self.lr_scheduler.step()
                    self._total_updates += 1
                    self._total_updates_made += 1

        # Remaining gradient steps
        if n_updates % self._grad_accum_steps != 0:
            self._apply_grad_clipping()
            opt.step()
            opt.zero_grad()
            self.update_ema()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            self._total_updates += 1
            self._total_updates_made += 1

        return {
            "policy_loss": total_policy_loss / max(n_updates, 1),
            "value_loss": total_value_loss / max(n_updates, 1),
            "entropy": total_entropy / max(n_updates, 1),
            "approx_kl": total_approx_kl / max(n_updates, 1),
            "explained_var": total_explained_var / max(n_updates, 1),
        }

    def _apply_grad_clipping(self) -> None:
        # Feature 28-29: Separate actor/critic grad clipping
        if config.ACTOR_GRAD_CLIP_NORM > 0:
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), config.ACTOR_GRAD_CLIP_NORM)
        elif config.GRAD_CLIP_NORM > 0:
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), config.GRAD_CLIP_NORM)

        if config.CRITIC_GRAD_CLIP_NORM > 0:
            torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), config.CRITIC_GRAD_CLIP_NORM)
        elif config.GRAD_CLIP_NORM > 0:
            torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), config.GRAD_CLIP_NORM)

        if config.GRAD_CLIP_VALUE > 0:
            for p in list(self.policy.parameters()) + list(self.value_net.parameters()):
                if p.grad is not None:
                    p.grad.data.clamp_(-config.GRAD_CLIP_VALUE, config.GRAD_CLIP_VALUE)

    def reset_episode(self) -> None:
        if self._frame_stack_size > 1:
            self._frame_stack = deque(
                [torch.zeros(self.obs_dim)] * self._frame_stack_size,
                maxlen=self._frame_stack_size,
            )
