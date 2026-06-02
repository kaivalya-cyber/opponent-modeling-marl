"""Opponent-modeling agent — extends PPO with a GRU/Transformer opponent model."""

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
from models.opponent_model import build_opponent_model, soft_update_target
from training.rollout_buffer import RolloutBuffer


class OMAgent(BaseAgent):
    """PPO agent augmented with a learned opponent model."""

    def __init__(self, obs_dim: int = 29, action_dim: int = 5) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self._frame_stack_size = config.FRAME_STACK
        self._frame_stack: deque | None = None
        self._effective_obs_dim = obs_dim * self._frame_stack_size

        augmented_dim = obs_dim + action_dim
        self._effective_augmented_dim = augmented_dim

        self.policy = PolicyNet(input_dim=augmented_dim, action_dim=action_dim).to(config.DEVICE)
        self.value_net = ValueNet(input_dim=augmented_dim).to(config.DEVICE)
        self.opponent_model = build_opponent_model(
            input_dim=config.OM_INPUT_DIM,
            hidden_dim=config.OM_HIDDEN_DIM,
            action_dim=action_dim,
            model_type=config.OM_MODEL_TYPE,
        ).to(config.DEVICE)

        self.target_opponent_model = None
        if config.OM_USE_TARGET_NET:
            self.target_opponent_model = build_opponent_model(
                input_dim=config.OM_INPUT_DIM,
                hidden_dim=config.OM_HIDDEN_DIM,
                action_dim=action_dim,
                model_type=config.OM_MODEL_TYPE,
            ).to(config.DEVICE)
            self.target_opponent_model.load_state_dict(self.opponent_model.state_dict())
            for p in self.target_opponent_model.parameters():
                p.requires_grad = False

        self._om_num_layers = config.OM_NUM_LAYERS if config.OM_MODEL_TYPE == "gru" else 1
        self._om_hidden_dim = config.OM_HIDDEN_DIM

        self.h: torch.Tensor = torch.zeros(
            self._om_num_layers, 1, self._om_hidden_dim, device=config.DEVICE
        )

        # Feature 4-6: Separate optimizers
        self._use_separate_opt = config.USE_SEPARATE_OPTIMIZERS
        om_params = list(self.opponent_model.parameters())
        if self._use_separate_opt:
            self.policy_optimizer = torch.optim.Adam(
                list(self.policy.parameters()) + om_params,
                lr=config.POLICY_LR, weight_decay=config.POLICY_WEIGHT_DECAY,
            )
            self.value_optimizer = torch.optim.Adam(
                self.value_net.parameters(), lr=config.VALUE_LR,
                weight_decay=config.VALUE_WEIGHT_DECAY,
            )
            self.optimizer = self.policy_optimizer
        else:
            self.optimizer = torch.optim.Adam(
                list(self.policy.parameters())
                + list(self.value_net.parameters())
                + om_params,
                lr=config.LEARNING_RATE,
                weight_decay=config.POLICY_WEIGHT_DECAY,
            )

        # Feature 16-17: Model EMA
        self.init_ema()

        self._current_entropy_coeff = config.ENTROPY_COEFF
        self._current_clip_epsilon = config.CLIP_EPSILON
        self._total_updates_made = 0

        self.lr_scheduler = self._build_lr_scheduler()
        self._lr_warmup_steps = config.LR_WARMUP_STEPS
        self._total_updates = 0

        self._grad_accum_steps = config.GRAD_ACCUMULATION_STEPS

        self.om_loss_weight = config.OM_LOSS_WEIGHT

        # Feature 18: OM accuracy tracking
        self._om_correct = deque(maxlen=config.OM_ACCURACY_WINDOW)

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
                cosine = CosineAnnealingLR(self.optimizer, T_max=max(1, total_steps - config.LR_WARMUP_STEPS), eta_min=min_lr)
                return SequentialLR(self.optimizer, schedulers=[warmup, cosine], milestones=[config.LR_WARMUP_STEPS])
            else:
                return CosineAnnealingLR(self.optimizer, T_max=max(1, total_steps), eta_min=min_lr)
        elif config.LR_SCHEDULER_TYPE == "linear":
            if config.LR_WARMUP_STEPS > 0:
                warmup = LinearLR(self.optimizer, start_factor=0.01, end_factor=1.0, total_iters=config.LR_WARMUP_STEPS)
                linear = LinearLR(self.optimizer, start_factor=1.0, end_factor=config.LR_MIN_RATIO,
                                  total_iters=max(1, total_steps - config.LR_WARMUP_STEPS))
                return SequentialLR(self.optimizer, schedulers=[warmup, linear], milestones=[config.LR_WARMUP_STEPS])
            else:
                return LinearLR(self.optimizer, start_factor=1.0, end_factor=config.LR_MIN_RATIO, total_iters=max(1, total_steps))
        return None

    def _build_frame_stack_obs(self, obs: np.ndarray) -> np.ndarray:
        if self._frame_stack_size <= 1:
            return obs
        obs_t = torch.tensor(obs, dtype=torch.float32)
        self._frame_stack.append(obs_t)
        stacked = torch.cat(list(self._frame_stack), dim=-1).numpy()
        return stacked

    def get_augmented_obs(self, own_obs: np.ndarray, last_opp_action: int) -> torch.Tensor:
        own_obs = self.normalize_obs(own_obs)
        self.update_obs_normalizer(own_obs)

        opp_action_onehot = F.one_hot(
            torch.tensor(last_opp_action, device=config.DEVICE),
            num_classes=self.action_dim,
        ).float()
        obs_t = torch.tensor(own_obs, dtype=torch.float32, device=config.DEVICE)

        gru_input = torch.cat([opp_action_onehot, obs_t]).unsqueeze(0).unsqueeze(0)

        om_model = self.target_opponent_model if self.target_opponent_model is not None else self.opponent_model

        # Feature 14: MC Dropout for OM uncertainty
        if config.OM_MC_DROPOUT_SAMPLES > 0 and self.training:
            om_model.train()
            mc_probs = []
            for _ in range(config.OM_MC_DROPOUT_SAMPLES):
                probs, new_h, _ = om_model.predict_opponent_action(gru_input, self.h)
                mc_probs.append(probs.unsqueeze(0))
            opp_probs = torch.mean(torch.cat(mc_probs, dim=0), dim=0)
            om_model.eval()
        else:
            with torch.no_grad():
                opp_probs, new_h, confidence = om_model.predict_opponent_action(gru_input, self.h)

        self.h = new_h.detach()

        if config.OM_CONFIDENCE_THRESHOLD > 0 and confidence is not None:
            conf = confidence.squeeze(0).item()
            if conf < config.OM_CONFIDENCE_THRESHOLD:
                opp_probs = torch.ones_like(opp_probs) / self.action_dim

        opp_probs = opp_probs.squeeze(0)
        augmented = torch.cat([obs_t, opp_probs])
        return augmented.cpu()

    def select_action(self, obs: np.ndarray, *, last_opp_action: int = 0, **kwargs) -> tuple[int, torch.Tensor, torch.Tensor]:
        augmented = self.get_augmented_obs(obs, last_opp_action)
        aug_t = augmented.unsqueeze(0).to(config.DEVICE)

        with torch.no_grad():
            logits = self.policy(aug_t, temperature=config.SOFTMAX_TEMPERATURE)
            value = self.value_net(aug_t)

        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        action_int = action.item()
        action_int = self.add_action_noise(action_int, self.action_dim)

        return action_int, log_prob.squeeze(0).cpu(), value.squeeze(0).cpu()

    def select_action_deterministic(self, obs: np.ndarray, *, last_opp_action: int = 0, **kwargs) -> int:
        augmented = self.get_augmented_obs(obs, last_opp_action)
        aug_t = augmented.unsqueeze(0).to(config.DEVICE)

        with torch.no_grad():
            logits = self.policy(aug_t)
        return int(logits.argmax(dim=-1).item())

    def update(self, rollout_buffer: RolloutBuffer) -> dict:
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_om_loss = 0.0
        total_approx_kl = 0.0
        total_explained_var = 0.0
        total_om_accuracy = 0.0
        n_updates = 0

        if config.ENTROPY_SCHEDULE_STEPS > 0 and self._total_updates_made < config.ENTROPY_SCHEDULE_STEPS:
            progress = self._total_updates_made / config.ENTROPY_SCHEDULE_STEPS
            self._current_entropy_coeff = config.ENTROPY_COEFF * (1.0 - progress) + config.ENTROPY_TARGET * progress

        if config.CLIP_EPSILON_SCHEDULE_STEPS > 0:
            progress = min(1.0, self._total_updates_made / config.CLIP_EPSILON_SCHEDULE_STEPS)
            self._current_clip_epsilon = config.CLIP_EPSILON * (1.0 - progress) + 0.05 * progress

        # Feature 15: OM weight ramp-up
        ramp = min(1.0, self._total_updates_made / max(1, config.OM_WEIGHT_RAMP_EPISODES)) if config.OM_WEIGHT_RAMP_EPISODES > 0 else 1.0

        opt = self.optimizer
        opt.zero_grad()

        for epoch in range(config.N_EPOCHS):
            for batch_idx, batch in enumerate(rollout_buffer.get_batches(config.MINIBATCH_SIZE)):
                stored_obs = batch["obs"]
                actions = batch["actions"]
                old_log_probs = batch["log_probs"]
                advantages = batch["advantages"]
                returns = batch["returns"]
                opp_actions = batch["opp_actions"]

                raw_obs = stored_obs[:, :self.obs_dim]

                # ---- Opponent model ----
                opp_onehot = F.one_hot(opp_actions, num_classes=self.action_dim).float()
                gru_in = torch.cat([opp_onehot, raw_obs], dim=-1).unsqueeze(1)

                om_logits, _ = self.opponent_model(gru_in)
                opp_probs = F.softmax(om_logits, dim=-1)

                # Feature 12: Focal loss for OM
                if config.OM_FOCAL_LOSS_GAMMA > 0:
                    ce_loss = F.cross_entropy(om_logits, opp_actions, reduction="none")
                    pt = torch.exp(-ce_loss)
                    om_loss_raw = (1 - pt) ** config.OM_FOCAL_LOSS_GAMMA * ce_loss
                else:
                    om_loss_raw = F.cross_entropy(om_logits, opp_actions, reduction="none")

                # Feature 13: Label smoothing for OM
                if config.OM_LABEL_SMOOTHING > 0:
                    n_classes = self.action_dim
                    smooth_targets = opp_onehot * (1 - config.OM_LABEL_SMOOTHING) + config.OM_LABEL_SMOOTHING / n_classes
                    om_loss_smooth = -torch.sum(smooth_targets * F.log_softmax(om_logits, dim=-1), dim=-1)
                    om_loss_raw = config.OM_LABEL_SMOOTHING * om_loss_smooth + (1 - config.OM_LABEL_SMOOTHING) * om_loss_raw

                # Feature 11: OM accuracy tracking
                if config.OM_LOG_ACCURACY:
                    pred_actions = om_logits.argmax(dim=-1)
                    accuracy = (pred_actions == opp_actions).float().mean().item()
                    total_om_accuracy += accuracy

                if config.OM_SURPRISE_THRESHOLD > 0:
                    surprise_mask = (om_loss_raw > config.OM_SURPRISE_THRESHOLD).float()
                    om_loss = (om_loss_raw * surprise_mask).mean()
                else:
                    om_loss = om_loss_raw.mean()

                if config.OM_CLIP_LOSS > 0:
                    om_loss = torch.clamp(om_loss, max=config.OM_CLIP_LOSS)

                aug_obs = torch.cat([raw_obs, opp_probs], dim=-1)

                # ---- PPO losses ----
                logits = self.policy(aug_obs, temperature=config.SOFTMAX_TEMPERATURE)
                values = self.value_net(aug_obs).squeeze(-1)

                if config.VALUE_RESIDUAL_CLIP > 0:
                    values = returns + torch.clamp(values - returns, -config.VALUE_RESIDUAL_CLIP, config.VALUE_RESIDUAL_CLIP)

                adv = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                dist = Categorical(logits=logits)
                new_log_probs = dist.log_prob(actions)
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_log_probs - old_log_probs)
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1.0 - self._current_clip_epsilon, 1.0 + self._current_clip_epsilon) * adv
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(values, returns)

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - (new_log_probs - old_log_probs)).mean().item()
                    var_explained = 1.0 - (F.mse_loss(values, returns).item() / (returns.var() + 1e-8).item())
                    total_approx_kl += approx_kl
                    total_explained_var += var_explained

                loss = (
                    policy_loss
                    + config.VALUE_LOSS_COEFF * value_loss
                    - self._current_entropy_coeff * entropy
                    + self.om_loss_weight * ramp * om_loss
                )

                loss = loss / self._grad_accum_steps
                loss.backward()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                total_om_loss += om_loss.item()
                n_updates += 1

                if (batch_idx + 1) % self._grad_accum_steps == 0:
                    self._apply_grad_clipping()
                    opt.step()
                    opt.zero_grad()

                    if config.OM_USE_TARGET_NET and self.target_opponent_model is not None:
                        if config.OM_TARGET_UPDATE_INTERVAL > 0 and self._total_updates % config.OM_TARGET_UPDATE_INTERVAL == 0:
                            self.target_opponent_model.load_state_dict(self.opponent_model.state_dict())
                        else:
                            soft_update_target(self.target_opponent_model, self.opponent_model, config.OM_TARGET_TAU)

                    self.update_ema()
                    if self.lr_scheduler is not None:
                        self.lr_scheduler.step()
                    self._total_updates += 1
                    self._total_updates_made += 1

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
            "om_loss": total_om_loss / max(n_updates, 1),
            "approx_kl": total_approx_kl / max(n_updates, 1),
            "explained_var": total_explained_var / max(n_updates, 1),
            "om_accuracy": total_om_accuracy / max(n_updates, 1) if config.OM_LOG_ACCURACY else 0.0,
        }

    def _apply_grad_clipping(self) -> None:
        all_params = list(self.policy.parameters()) + list(self.value_net.parameters()) + list(self.opponent_model.parameters())
        if config.ACTOR_GRAD_CLIP_NORM > 0:
            torch.nn.utils.clip_grad_norm_(list(self.policy.parameters()) + list(self.opponent_model.parameters()), config.ACTOR_GRAD_CLIP_NORM)
        elif config.GRAD_CLIP_NORM > 0:
            torch.nn.utils.clip_grad_norm_(all_params, config.GRAD_CLIP_NORM)

        if config.CRITIC_GRAD_CLIP_NORM > 0:
            torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), config.CRITIC_GRAD_CLIP_NORM)

        if config.GRAD_CLIP_VALUE > 0:
            for p in all_params:
                if p.grad is not None:
                    p.grad.data.clamp_(-config.GRAD_CLIP_VALUE, config.GRAD_CLIP_VALUE)

    def reset_episode(self) -> None:
        self.h = torch.zeros(self._om_num_layers, 1, self._om_hidden_dim, device=config.DEVICE)
        if self._frame_stack_size > 1:
            self._frame_stack = deque([torch.zeros(self.obs_dim)] * self._frame_stack_size, maxlen=self._frame_stack_size)
