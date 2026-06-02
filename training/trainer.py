"""
Training loop for self-play multi-agent RL.

1. Initialize env, agent_predator, agent_prey (same class for self-play)
2. Curriculum learning: progressive grid sizes (5→7→10)
3. Collect ROLLOUT_STEPS transitions, then run PPO update for both agents
4. Log metrics to CSV every episode + wandb + TensorBoard (optional)
5. Run evaluation against random opponent every EVAL_INTERVAL episodes
6. Run past-self evaluation (pit current agent vs historical snapshot)
7. Save checkpoints every CHECKPOINT_INTERVAL episodes
"""

from __future__ import annotations

import copy
import csv
import json
import logging
import os
import platform
import subprocess
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

import config
from env.predator_prey import PredatorPreyEnv
from agents.base_agent import BaseAgent
from training.rollout_buffer import RolloutBuffer
from training.elo import EloRating

logger = logging.getLogger(__name__)

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


class CheckpointManager:
    """Manages saving top K checkpoints based on a metric."""

    def __init__(self, results_dir: Path, keep_best: int = 5, metric: str = "capture_rate") -> None:
        self.results_dir = results_dir
        self.keep_best = max(keep_best, 0)
        self.metric = metric
        self.best_metrics: list[tuple[float, str]] = []

    def save(self, state: dict, current_metric: float, tag: str) -> None:
        if self.keep_best <= 0:
            path = self.results_dir / f"checkpoint_{tag}.pt"
            torch.save(state, path)
            return

        self.best_metrics.append((current_metric, tag))
        self.best_metrics.sort(key=lambda x: x[0], reverse=True)

        if len(self.best_metrics) > self.keep_best:
            _, remove_tag = self.best_metrics.pop(-1)
            old_path = self.results_dir / f"checkpoint_{remove_tag}.pt"
            if old_path.exists():
                old_path.unlink()

        path = self.results_dir / f"checkpoint_{tag}.pt"
        torch.save(state, path)
        logger.info(
            f"Checkpoint saved: {path} (metric={current_metric:.4f}, "
            f"best={self.best_metrics[0][0]:.4f})"
        )


class Trainer:
    """Self-play trainer for two agents in PredatorPreyEnv."""

    def __init__(
        self,
        predator: BaseAgent,
        prey: BaseAgent,
        experiment_name: str,
        total_episodes: int = config.TOTAL_EPISODES,
        is_om: bool = False,
    ) -> None:
        self.env = PredatorPreyEnv(
            grid_size=config.GRID_SIZE, max_steps=config.MAX_STEPS
        )
        self.predator = predator
        self.prey = prey
        self.experiment_name = experiment_name
        self.total_episodes = total_episodes
        self.is_om = is_om

        self.buf_obs_dim = config.OBS_DIM + config.ACTION_DIM if is_om else config.OBS_DIM

        self.elo = EloRating()

        self.results_dir = Path("results") / experiment_name
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.snapshots_dir = self.results_dir / "snapshots"
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)

        # Wandb initialization
        self._use_wandb = config.USE_WANDB and WANDB_AVAILABLE
        if self._use_wandb:
            wandb.init(
                project=config.WANDB_PROJECT,
                name=experiment_name,
                config=self._make_wandb_config(),
            )
        elif config.USE_WANDB and not WANDB_AVAILABLE:
            logger.warning("wandb not installed — set USE_WANDB=False or pip install wandb")

        # --- NEW: TensorBoard ---
        self._use_tb = config.USE_TENSORBOARD and TENSORBOARD_AVAILABLE
        self._tb_writer = None
        if self._use_tb:
            tb_dir = Path(config.TENSORBOARD_DIR) / experiment_name
            tb_dir.mkdir(parents=True, exist_ok=True)
            self._tb_writer = SummaryWriter(log_dir=str(tb_dir))
        elif config.USE_TENSORBOARD and not TENSORBOARD_AVAILABLE:
            logger.warning("TensorBoard not available — install with: pip install tensorboard")

        # Curriculum learning tracking
        self._curriculum_enabled = config.CURRICULUM_ENABLED
        self._curriculum_stage = 0

        # CSV logging
        self.csv_path = self.results_dir / "metrics.csv"
        self.eval_csv_path = self.results_dir / "eval_metrics.csv"
        self._init_csv()
        self._init_eval_csv()

        self.recent_captures = deque(maxlen=20)

        # --- NEW: Checkpoint manager ---
        self._ckpt_mgr = CheckpointManager(
            self.results_dir,
            keep_best=config.CKPT_KEEP_BEST,
            metric=config.CKPT_METRIC,
        )

        # --- NEW: Action repeat ---
        self._action_repeat = config.ACTION_REPEAT

        # Save run metadata
        self._save_run_metadata()

    def _make_wandb_config(self) -> dict:
        return {
            "is_om": self.is_om,
            "grid_size": config.GRID_SIZE,
            "total_episodes": self.total_episodes,
            "learning_rate": config.LEARNING_RATE,
            "curriculum_enabled": config.CURRICULUM_ENABLED,
            "lr_scheduler": config.LR_SCHEDULER_TYPE,
            "lr_warmup": config.LR_WARMUP_STEPS,
            "grad_accumulation": config.GRAD_ACCUMULATION_STEPS,
            "n_step_returns": config.N_STEP_RETURNS,
            "obs_normalize": config.OBS_NORMALIZE,
            "reward_normalize": config.REWARD_NORMALIZE,
            "om_target_net": config.OM_USE_TARGET_NET,
            "om_ensemble_size": config.OM_ENSEMBLE_SIZE,
            "om_confidence_threshold": config.OM_CONFIDENCE_THRESHOLD,
            "om_surprise_threshold": config.OM_SURPRISE_THRESHOLD,
            "icm_enabled": config.ICM_ENABLED,
            "param_noise": config.PARAM_NOISE_ENABLED,
            "action_repeat": config.ACTION_REPEAT,
            "frame_stack": config.FRAME_STACK,
        }

    def _init_csv(self) -> None:
        fieldnames = [
            "episode",
            "predator_reward",
            "prey_reward",
            "capture_rate",
            "predator_elo",
            "prey_elo",
            "policy_loss",
            "value_loss",
            "entropy",
        ]
        if self.is_om:
            fieldnames.append("om_loss")
            fieldnames.append("om_loss_weight")
        if self._curriculum_enabled:
            fieldnames.append("grid_size")
        # --- NEW: extra fields ---
        if config.LR_SCHEDULER_TYPE != "none":
            fieldnames.append("learning_rate")
        if config.ICM_ENABLED:
            fieldnames.append("icm_fwd_loss")
            fieldnames.append("icm_inv_loss")
        if config.PARAM_NOISE_ENABLED:
            fieldnames.append("param_noise_std")
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

    def _init_eval_csv(self) -> None:
        fieldnames = [
            "episode",
            "win_rate_vs_random",
            "avg_episode_length",
            "avg_predator_reward",
        ]
        if config.PAST_SELF_EVAL:
            fieldnames.append("win_rate_vs_past_self")
        with open(self.eval_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

    def _log_row(self, row: dict) -> None:
        fieldnames = list(row.keys())
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow(row)

    def _log_eval_row(self, row: dict) -> None:
        fieldnames = list(row.keys())
        with open(self.eval_csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow(row)

    def _get_curriculum_params(self, episode: int) -> tuple[int, int]:
        if not self._curriculum_enabled:
            return config.GRID_SIZE, config.MAX_STEPS

        schedule = config.CURRICULUM_SCHEDULE
        while (
            self._curriculum_stage + 1 < len(schedule)
            and episode >= schedule[self._curriculum_stage + 1][0]
        ):
            self._curriculum_stage += 1

        _, grid_size, max_steps = schedule[self._curriculum_stage]
        return grid_size, max_steps

    def _ensure_env_matches_curriculum(self, episode: int) -> bool:
        new_grid_size, new_max_steps = self._get_curriculum_params(episode)
        if (
            self.env.grid_size != new_grid_size
            or self.env.max_steps != new_max_steps
        ):
            logger.info(
                f"Curriculum: advancing to grid={new_grid_size}, "
                f"max_steps={new_max_steps} @ episode {episode}"
            )
            self.env = PredatorPreyEnv(
                grid_size=new_grid_size, max_steps=new_max_steps,
            )
            if self._use_wandb:
                wandb.log({"curriculum/grid_size": new_grid_size, "episode": episode})
            if self._tb_writer is not None:
                self._tb_writer.add_scalar("curriculum/grid_size", new_grid_size, episode)
            return True
        return False

    def train(self) -> None:
        """Run the full self-play training loop."""
        pred_buf = RolloutBuffer(
            obs_dim=self.buf_obs_dim, action_dim=config.ACTION_DIM,
            capacity=config.ROLLOUT_STEPS,
        )
        prey_buf = RolloutBuffer(
            obs_dim=self.buf_obs_dim, action_dim=config.ACTION_DIM,
            capacity=config.ROLLOUT_STEPS,
        )

        # Feature 34: Streak tracking
        predator_streak = 0
        prey_streak = 0
        best_streak = 0

        # Feature 35: Best episode tracking
        best_capture_rate = 0.0
        best_ep_reward_pred = -float("inf")
        best_ep_reward_prey = -float("inf")

        # Feature 37-38: Speed and ETA
        train_start_time = time.time()
        last_speed_log_time = time.time()

        # Feature 39: Config diff
        if config.LOG_CONFIG_DIFF:
            self._log_config_diff()

        # Feature 48: Profile mode
        profile_timings: dict[str, list[float]] = {}

        global_step = 0
        episode = 0
        pred_metrics: dict = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
        if self.is_om:
            pred_metrics["om_loss"] = 0.0

        # Feature 19-20: Early stopping vars
        no_improve_episodes = 0

        # Progress bar
        pbar = None
        if TQDM_AVAILABLE:
            pbar = tqdm(total=self.total_episodes, desc=self.experiment_name, unit="ep")

        while episode < self.total_episodes:
            if self._ensure_env_matches_curriculum(episode):
                if config.PAST_SELF_EVAL and episode >= config.SNAPSHOT_INTERVAL:
                    logger.info(
                        "Curriculum grid changed while past-self eval is active. "
                        "Past snapshots were trained on smaller grids — "
                        "past-self win rates may be inflated."
                    )

            # --- NEW: Parameter noise ---
            if config.PARAM_NOISE_ENABLED:
                self.predator._apply_param_noise()
                self.prey._apply_param_noise()

            (obs_pred, obs_prey), _ = self.env.reset()
            self.predator.reset_episode()
            self.prey.reset_episode()

            last_pred_action = 0
            last_prey_action = 0
            ep_reward_pred = 0.0
            ep_reward_prey = 0.0
            done = False

            # --- NEW: Action repeat tracking ---
            repeat_counter = 0
            repeated_a_pred = 0
            repeated_a_prey = 0

            while not done:
                # --- NEW: Action repeat ---
                if repeat_counter <= 0 or self._action_repeat <= 1:
                    if self.is_om:
                        aug_pred_t = self.predator.get_augmented_obs(obs_pred, last_prey_action)
                        a_pred, lp_pred, v_pred = self.predator.select_action(
                            obs_pred, last_opp_action=last_prey_action
                        )
                        aug_prey_t = self.prey.get_augmented_obs(obs_prey, last_pred_action)
                        a_prey, lp_prey, v_prey = self.prey.select_action(
                            obs_prey, last_opp_action=last_pred_action
                        )
                        aug_pred = aug_pred_t.numpy()
                        aug_prey = aug_prey_t.numpy()
                    else:
                        a_pred, lp_pred, v_pred = self.predator.select_action(obs_pred)
                        a_prey, lp_prey, v_prey = self.prey.select_action(obs_prey)
                        aug_pred = obs_pred
                        aug_prey = obs_prey

                    repeated_a_pred = a_pred
                    repeated_a_prey = a_prey
                    repeat_counter = self._action_repeat
                else:
                    a_pred = repeated_a_pred
                    a_prey = repeated_a_prey
                    lp_pred, v_pred = torch.tensor(0.0), torch.tensor(0.0)
                    lp_prey, v_prey = torch.tensor(0.0), torch.tensor(0.0)
                    if self.is_om:
                        aug_pred = self.predator.get_augmented_obs(obs_pred, last_prey_action).numpy()
                        aug_prey = self.prey.get_augmented_obs(obs_prey, last_pred_action).numpy()
                    else:
                        aug_pred = obs_pred
                        aug_prey = obs_prey

                repeat_counter -= 1

                # Step environment
                (next_obs_pred, next_obs_prey), (r_pred, r_prey), terminated, truncated, info = (
                    self.env.step((a_pred, a_prey))
                )
                done = terminated or truncated

                # --- NEW: Intrinsic curiosity reward ---
                if config.ICM_ENABLED:
                    icm_bonus_pred = self.predator.get_icm_intrinsic_reward(obs_pred, next_obs_pred, a_pred)
                    icm_bonus_prey = self.prey.get_icm_intrinsic_reward(obs_prey, next_obs_prey, a_prey)
                    r_pred += icm_bonus_pred
                    r_prey += icm_bonus_prey

                # --- NEW: Reward normalization ---
                r_pred = self.predator.normalize_reward(r_pred)
                r_prey = self.prey.normalize_reward(r_prey)

                # Store transitions
                store_obs_pred = aug_pred
                store_obs_prey = aug_prey

                if not pred_buf.full:
                    pred_buf.add(
                        obs=store_obs_pred, action=a_pred, reward=r_pred,
                        value=v_pred, log_prob=lp_pred, done=done,
                        opp_action=a_prey,
                    )
                if not prey_buf.full:
                    prey_buf.add(
                        obs=store_obs_prey, action=a_prey, reward=r_prey,
                        value=v_prey, log_prob=lp_prey, done=done,
                        opp_action=a_pred,
                    )

                obs_pred, obs_prey = next_obs_pred, next_obs_prey
                last_pred_action, last_prey_action = a_pred, a_prey
                ep_reward_pred += r_pred
                ep_reward_prey += r_prey
                global_step += 1

                if pred_buf.full and prey_buf.full:
                    if self.is_om:
                        _, _, last_v_pred = self.predator.select_action(
                            obs_pred, last_opp_action=last_prey_action
                        )
                        _, _, last_v_prey = self.prey.select_action(
                            obs_prey, last_opp_action=last_pred_action
                        )
                    else:
                        _, _, last_v_pred = self.predator.select_action(obs_pred)
                        _, _, last_v_prey = self.prey.select_action(obs_prey)

                    pred_buf.compute_returns(last_v_pred, config.GAMMA, config.GAE_LAMBDA)
                    prey_buf.compute_returns(last_v_prey, config.GAMMA, config.GAE_LAMBDA)

                    pred_metrics = self.predator.update(pred_buf)
                    prey_metrics = self.prey.update(prey_buf)

                    # --- NEW: ICM update ---
                    if config.ICM_ENABLED:
                        for batch in pred_buf.get_batches(config.MINIBATCH_SIZE):
                            fwd, inv = self.predator.icm_update(batch)
                            break  # single batch per update
                        pred_metrics["icm_fwd_loss"] = fwd
                        pred_metrics["icm_inv_loss"] = inv

                    pred_buf.reset()
                    prey_buf.reset()

            # --- NEW: Remove parameter noise after episode ---
            if config.PARAM_NOISE_ENABLED:
                self.predator._remove_param_noise()
                self.prey._remove_param_noise()
                self.predator.decay_param_noise()
                self.prey.decay_param_noise()

            captured = info["captured"]
            self.recent_captures.append(1.0 if captured else 0.0)

            if captured:
                self.elo.update("predator", "prey")
            else:
                self.elo.draw("predator", "prey")

            capture_rate = sum(self.recent_captures) / len(self.recent_captures)

            if self.is_om:
                self.predator.om_loss_weight *= config.OM_LOSS_WEIGHT_DECAY
                self.prey.om_loss_weight *= config.OM_LOSS_WEIGHT_DECAY

            pol_loss = round(pred_metrics["policy_loss"], 6)
            val_loss = round(pred_metrics["value_loss"], 6)
            ent = round(pred_metrics["entropy"], 6)

            # Feature 34: Streak tracking
            if captured:
                predator_streak += 1
                prey_streak = 0
            else:
                prey_streak += 1
                predator_streak = 0
            best_streak = max(best_streak, predator_streak, prey_streak)

            # Feature 35: Best episode tracking
            if capture_rate > best_capture_rate:
                best_capture_rate = capture_rate
            if ep_reward_pred > best_ep_reward_pred:
                best_ep_reward_pred = ep_reward_pred
            if ep_reward_prey > best_ep_reward_prey:
                best_ep_reward_prey = ep_reward_prey

            # Feature 36: Steps/sec
            now = time.time()
            elapsed = now - train_start_time
            steps_per_sec = global_step / elapsed if elapsed > 0 else 0.0

            # Feature 38: ETA
            if episode > 0:
                eps_per_sec = episode / elapsed
                eta_seconds = (self.total_episodes - episode) / eps_per_sec if eps_per_sec > 0 else 0.0
            else:
                eta_seconds = 0.0

            # Feature 33: Episode stats (from agent tracking)
            pred_ep_stats = self.predator.get_episode_stats()
            prey_ep_stats = self.prey.get_episode_stats()

            # Feature 37: Log speed periodically
            if config.LOG_STEPS_PER_SECOND and now - last_speed_log_time > 30.0:
                logger.info(f"Speed: {steps_per_sec:.0f} steps/sec, ETA: {eta_seconds / 60:.1f} min")
                last_speed_log_time = now

            # Feature 41: Time-limit bootstrap (truncated episode handling)
            time_limit_bootstrapped = 0.0
            if config.HANDLE_TIME_LIMIT_BOOTSTRAP and truncated and not terminated:
                time_limit_bootstrapped = 1.0

            row = {
                "episode": episode,
                "predator_reward": round(ep_reward_pred, 4),
                "prey_reward": round(ep_reward_prey, 4),
                "capture_rate": round(capture_rate, 4),
                "predator_elo": round(self.elo.get_rating("predator"), 2),
                "prey_elo": round(self.elo.get_rating("prey"), 2),
                "policy_loss": pol_loss,
                "value_loss": val_loss,
                "entropy": ent,
            }
            if self.is_om:
                row["om_loss"] = round(pred_metrics.get("om_loss", 0.0), 6)
                row["om_loss_weight"] = round(self.predator.om_loss_weight, 6)
                if config.OM_LOG_ACCURACY:
                    row["om_accuracy"] = round(self.predator.get_om_accuracy(), 4)
            if self._curriculum_enabled:
                row["grid_size"] = self.env.grid_size
            if config.LR_SCHEDULER_TYPE != "none":
                row["learning_rate"] = round(self.predator.optimizer.param_groups[0]["lr"], 8)
            if config.ICM_ENABLED:
                row["icm_fwd_loss"] = round(pred_metrics.get("icm_fwd_loss", 0.0), 6)
                row["icm_inv_loss"] = round(pred_metrics.get("icm_inv_loss", 0.0), 6)
            if config.PARAM_NOISE_ENABLED:
                row["param_noise_std"] = round(self.predator._param_noise_std, 6)
            if config.KL_TRACKING:
                row["approx_kl"] = round(pred_metrics.get("approx_kl", 0.0), 8)
            if config.EXPLAINED_VAR_TRACKING:
                row["explained_variance"] = round(pred_metrics.get("explained_variance", 0.0), 6)
            if config.ENTROPY_TARGET > 0.0 and config.ENTROPY_SCHEDULE_STEPS > 0:
                row["entropy_coeff"] = round(self.predator.get_entropy_coeff(), 8) if hasattr(self.predator, 'get_entropy_coeff') else 0.0
            if config.TRACK_STREAKS:
                row["predator_streak"] = predator_streak
                row["prey_streak"] = prey_streak
                row["best_streak"] = best_streak
            if config.TRACK_BEST_EPISODE:
                row["best_capture_rate"] = round(best_capture_rate, 4)
            if config.LOG_EPISODE_STATS:
                row["pred_ep_len"] = pred_ep_stats.get("episode_length", 0)
                row["pred_ep_reward"] = round(pred_ep_stats.get("episode_reward", 0.0), 4)
            if config.LOG_STEPS_PER_SECOND:
                row["steps_per_sec"] = round(steps_per_sec, 1)
            if config.LOG_ETA:
                row["eta_seconds"] = int(eta_seconds)
            if config.HANDLE_TIME_LIMIT_BOOTSTRAP:
                row["time_limit_boostrapped"] = time_limit_bootstrapped
            self._log_row(row)

            # Wandb logging
            if self._use_wandb:
                wandb_log = {
                    "episode": episode,
                    "capture_rate": capture_rate,
                    "predator_elo": self.elo.get_rating("predator"),
                    "prey_elo": self.elo.get_rating("prey"),
                    "predator_reward": ep_reward_pred,
                    "prey_reward": ep_reward_prey,
                    "policy_loss": pol_loss,
                    "value_loss": val_loss,
                    "entropy": ent,
                }
                if self.is_om:
                    wandb_log["om_loss"] = pred_metrics.get("om_loss", 0.0)
                    wandb_log["om_loss_weight"] = self.predator.om_loss_weight
                    if config.OM_LOG_ACCURACY:
                        wandb_log["om_accuracy"] = self.predator.get_om_accuracy()
                if self._curriculum_enabled:
                    wandb_log["grid_size"] = self.env.grid_size
                if config.LR_SCHEDULER_TYPE != "none":
                    wandb_log["learning_rate"] = self.predator.optimizer.param_groups[0]["lr"]
                if config.ICM_ENABLED:
                    wandb_log["icm_fwd_loss"] = pred_metrics.get("icm_fwd_loss", 0.0)
                    wandb_log["icm_inv_loss"] = pred_metrics.get("icm_inv_loss", 0.0)
                if config.PARAM_NOISE_ENABLED:
                    wandb_log["param_noise_std"] = self.predator._param_noise_std
                if config.KL_TRACKING:
                    wandb_log["approx_kl"] = pred_metrics.get("approx_kl", 0.0)
                if config.EXPLAINED_VAR_TRACKING:
                    wandb_log["explained_variance"] = pred_metrics.get("explained_variance", 0.0)
                if config.TRACK_STREAKS:
                    wandb_log["predator_streak"] = predator_streak
                    wandb_log["prey_streak"] = prey_streak
                if config.TRACK_BEST_EPISODE:
                    wandb_log["best_capture_rate"] = best_capture_rate
                if config.LOG_STEPS_PER_SECOND:
                    wandb_log["steps_per_sec"] = steps_per_sec
                if config.HANDLE_TIME_LIMIT_BOOTSTRAP:
                    wandb_log["time_limit_bootstrapped"] = time_limit_bootstrapped
                if config.EXPERIMENT_TAGS:
                    wandb_log["experiment_tags"] = ",".join(config.EXPERIMENT_TAGS)
                wandb.log(wandb_log, step=episode)

            # --- NEW: TensorBoard logging ---
            if self._tb_writer is not None:
                self._tb_writer.add_scalar("train/capture_rate", capture_rate, episode)
                self._tb_writer.add_scalar("train/predator_elo", self.elo.get_rating("predator"), episode)
                self._tb_writer.add_scalar("train/prey_elo", self.elo.get_rating("prey"), episode)
                self._tb_writer.add_scalar("train/predator_reward", ep_reward_pred, episode)
                self._tb_writer.add_scalar("train/prey_reward", ep_reward_prey, episode)
                self._tb_writer.add_scalar("train/policy_loss", pol_loss, episode)
                self._tb_writer.add_scalar("train/value_loss", val_loss, episode)
                self._tb_writer.add_scalar("train/entropy", ent, episode)
                if self.is_om:
                    self._tb_writer.add_scalar("train/om_loss", pred_metrics.get("om_loss", 0.0), episode)
                    self._tb_writer.add_scalar("train/om_loss_weight", self.predator.om_loss_weight, episode)
                    if config.OM_LOG_ACCURACY:
                        self._tb_writer.add_scalar("train/om_accuracy", self.predator.get_om_accuracy(), episode)
                if config.LR_SCHEDULER_TYPE != "none":
                    self._tb_writer.add_scalar("train/learning_rate",
                                               self.predator.optimizer.param_groups[0]["lr"], episode)
                if config.ICM_ENABLED:
                    self._tb_writer.add_scalar("train/icm_fwd_loss", pred_metrics.get("icm_fwd_loss", 0.0), episode)
                    self._tb_writer.add_scalar("train/icm_inv_loss", pred_metrics.get("icm_inv_loss", 0.0), episode)
                if config.PARAM_NOISE_ENABLED:
                    self._tb_writer.add_scalar("train/param_noise_std", self.predator._param_noise_std, episode)
                if config.KL_TRACKING:
                    self._tb_writer.add_scalar("train/approx_kl", pred_metrics.get("approx_kl", 0.0), episode)
                if config.EXPLAINED_VAR_TRACKING:
                    self._tb_writer.add_scalar("train/explained_variance", pred_metrics.get("explained_variance", 0.0), episode)
                if config.TRACK_STREAKS:
                    self._tb_writer.add_scalar("train/predator_streak", predator_streak, episode)
                    self._tb_writer.add_scalar("train/prey_streak", prey_streak, episode)
                if config.TRACK_BEST_EPISODE:
                    self._tb_writer.add_scalar("train/best_capture_rate", best_capture_rate, episode)
                if config.LOG_STEPS_PER_SECOND:
                    self._tb_writer.add_scalar("train/steps_per_sec", steps_per_sec, episode)
                if config.HANDLE_TIME_LIMIT_BOOTSTRAP:
                    self._tb_writer.add_scalar("train/time_limit_bootstrapped", time_limit_bootstrapped, episode)
                self._tb_writer.flush()

            if episode % 50 == 0:
                log_msg = (
                    f"Ep {episode:>4d} | Cap rate: {capture_rate:.2f} | "
                    f"Pred Elo: {self.elo.get_rating('predator'):.0f} | "
                    f"Prey Elo: {self.elo.get_rating('prey'):.0f}"
                )
                if self.is_om:
                    log_msg += f" | OM w: {self.predator.om_loss_weight:.4f}"
                    if config.OM_LOG_ACCURACY:
                        log_msg += f" | OM acc: {self.predator.get_om_accuracy():.3f}"
                if self._curriculum_enabled:
                    log_msg += f" | grid: {self.env.grid_size}"
                if config.LR_SCHEDULER_TYPE != "none":
                    log_msg += f" | lr: {self.predator.optimizer.param_groups[0]['lr']:.2e}"
                if config.KL_TRACKING:
                    log_msg += f" | KL: {pred_metrics.get('approx_kl', 0.0):.2e}"
                if config.TRACK_STREAKS:
                    log_msg += f" | S({predator_streak}/{prey_streak})"
                if config.LOG_STEPS_PER_SECOND:
                    log_msg += f" | {steps_per_sec:.0f} sps"
                logger.info(log_msg)

            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix({
                    "capture": f"{capture_rate:.2f}",
                    "p_elo": f"{self.elo.get_rating('predator'):.0f}",
                })

            # Evaluation against random opponent
            if episode > 0 and episode % config.EVAL_INTERVAL == 0:
                eval_row = self._run_eval(episode)

                if self._use_wandb:
                    wandb_log_eval = {
                        "eval/win_rate_vs_random": eval_row["win_rate_vs_random"],
                        "eval/avg_episode_length": eval_row["avg_episode_length"],
                        "eval/avg_predator_reward": eval_row["avg_predator_reward"],
                    }
                    if config.EVAL_CONFIDENCE_INTERVAL > 0 and "win_rate_ci" in eval_row:
                        wandb_log_eval["eval/win_rate_ci"] = eval_row["win_rate_ci"]
                    wandb.log(wandb_log_eval, step=episode)

                if self._tb_writer is not None:
                    self._tb_writer.add_scalar("eval/win_rate_vs_random", eval_row["win_rate_vs_random"], episode)
                    self._tb_writer.add_scalar("eval/avg_episode_length", eval_row["avg_episode_length"], episode)
                    if config.EVAL_CONFIDENCE_INTERVAL > 0 and "win_rate_ci" in eval_row:
                        self._tb_writer.add_scalar("eval/win_rate_ci", eval_row["win_rate_ci"], episode)

                if config.PAST_SELF_EVAL and episode >= config.SNAPSHOT_INTERVAL:
                    past_win_rate = self._run_past_self_eval(episode)
                    eval_row["win_rate_vs_past_self"] = round(past_win_rate, 4)
                    if self._use_wandb:
                        wandb.log(
                            {"eval/win_rate_vs_past_self": past_win_rate},
                            step=episode,
                        )
                    if self._tb_writer is not None:
                        self._tb_writer.add_scalar("eval/win_rate_vs_past_self", past_win_rate, episode)

                self._log_eval_row(eval_row)

            # Save snapshot for past-self evaluation
            if (
                episode > 0
                and episode % config.SNAPSHOT_INTERVAL == 0
                and config.PAST_SELF_EVAL
            ):
                self._save_snapshot(episode)

            # Checkpoint with manager
            if episode > 0 and episode % config.CHECKPOINT_INTERVAL == 0:
                self._save_checkpoint(episode, capture_rate)

            # Feature 19-20: Early stopping
            if config.EARLY_STOP_PATIENCE > 0:
                if capture_rate >= config.EARLY_STOP_THRESHOLD:
                    no_improve_episodes = 0
                else:
                    no_improve_episodes += 1
                if no_improve_episodes >= config.EARLY_STOP_PATIENCE:
                    logger.info(
                        f"Early stopping triggered at episode {episode}: "
                        f"capture rate {capture_rate:.3f} < {config.EARLY_STOP_THRESHOLD} "
                        f"for {config.EARLY_STOP_PATIENCE} episodes"
                    )
                    if pbar is not None:
                        pbar.close()
                    break

            # Feature 38: JSON metrics export
            if config.EXPORT_JSON_METRICS and episode > 0 and episode % config.CHECKPOINT_INTERVAL == 0:
                self._export_json_metrics(episode, capture_rate)

            # Feature 48: Profile mode timing
            if config.PROFILE_MODE and episode > 0 and episode % config.CHECKPOINT_INTERVAL == 0:
                profile_timings.setdefault("steps_per_sec", []).append(steps_per_sec)
                profile_timings.setdefault("capture_rate", []).append(capture_rate)
                profile_timings.setdefault("policy_loss", []).append(pol_loss)

            episode += 1

        # Final checkpoint
        self._save_checkpoint("final", capture_rate if episode > 0 else 0.0)

        if pbar is not None:
            pbar.close()

        # Feature 38: Final JSON export
        if config.EXPORT_JSON_METRICS:
            self._export_json_metrics("final", capture_rate)

        # Feature 35: Log best episode
        if config.TRACK_BEST_EPISODE:
            logger.info(
                f"Training completed. Best capture rate: {best_capture_rate:.3f}, "
                f"Best pred reward: {best_ep_reward_pred:.2f}, "
                f"Best prey reward: {best_ep_reward_prey:.2f}"
            )

        # Feature 34: Log best streak
        if config.TRACK_STREAKS:
            logger.info(f"Best streak (predator or prey): {best_streak}")

        # Feature 37-38: Final speed summary
        if config.LOG_STEPS_PER_SECOND:
            total_elapsed = time.time() - train_start_time
            logger.info(
                f"Total time: {total_elapsed / 60:.1f} min, "
                f"Avg speed: {global_step / total_elapsed:.0f} steps/sec"
            )

        # Feature 48: Profile mode summary
        if config.PROFILE_MODE and profile_timings:
            avg_sps = sum(profile_timings["steps_per_sec"]) / len(profile_timings["steps_per_sec"])
            avg_cr = sum(profile_timings["capture_rate"]) / len(profile_timings["capture_rate"])
            logger.info(
                f"Profile: avg {avg_sps:.0f} steps/sec, "
                f"avg capture rate {avg_cr:.3f}"
            )

        # Feature 50: Experiment description
        if config.EXPERIMENT_DESCRIPTION:
            logger.info(f"Experiment: {config.EXPERIMENT_DESCRIPTION}")

        if self._use_wandb:
            wandb.finish()

        if self._tb_writer is not None:
            self._tb_writer.close()

        logger.info("Training complete.")

    # Feature 38: JSON export
    def _export_json_metrics(self, episode: int | str, capture_rate_override: float | None = None) -> None:
        import json
        metrics_path = os.path.join(str(self.results_dir), f"metrics_{episode}.json")
        cr = capture_rate_override if capture_rate_override is not None else 0.0
        metrics = {
            "episode": episode,
            "experiment_name": self.experiment_name,
            "capture_rate": cr,
            "predator_elo": self.elo.get_rating("predator"),
            "prey_elo": self.elo.get_rating("prey"),
        }
        if config.EXPERIMENT_TAGS:
            metrics["tags"] = list(config.EXPERIMENT_TAGS)
        if config.EXPERIMENT_DESCRIPTION:
            metrics["description"] = config.EXPERIMENT_DESCRIPTION
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

    # Feature 39: Log config diff from defaults
    def _log_config_diff(self) -> None:
        """Log differences between current config and default values."""
        default_config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.py")
        if not os.path.exists(default_config_path):
            logger.warning("Cannot log config diff: config.py not found")
            return
        try:
            import ast
            with open(default_config_path) as f:
                tree = ast.parse(f.read())
            defaults = {}
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id.isupper():
                            if isinstance(node.value, ast.Constant):
                                defaults[target.id] = node.value.value
                            elif isinstance(node.value, ast.List):
                                defaults[target.id] = [elt.value for elt in node.value.elts if isinstance(elt, ast.Constant)]
            diffs = []
            for key in dir(config):
                if key.isupper() and not key.startswith("_") and key in defaults:
                    current = getattr(config, key)
                    default = defaults[key]
                    if current != default:
                        diffs.append(f"{key}: {default!r} -> {current!r}")
            if diffs:
                logger.info(f"Config diffs from defaults ({len(diffs)}): " + "; ".join(diffs[:20]))
                if len(diffs) > 20:
                    logger.info(f"... and {len(diffs) - 20} more")
            else:
                logger.info("Config: all values match defaults")
        except Exception as e:
            logger.debug(f"Config diff failed: {e}")

    # ------------------------------------------------------------------
    # Run Metadata
    # ------------------------------------------------------------------
    def _save_run_metadata(self) -> None:
        git_hash = "unknown"
        git_branch = "unknown"
        try:
            git_hash = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True,
            ).strip()
            git_branch = subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True,
            ).strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass

        config_snapshot = {}
        for key in dir(config):
            if key.isupper() and not key.startswith("_"):
                value = getattr(config, key)
                if isinstance(value, (str, int, float, bool, type(None))):
                    config_snapshot[key] = value
                elif isinstance(value, (list, tuple)):
                    config_snapshot[key] = list(value)
                else:
                    config_snapshot[key] = str(value)

        hardware_info = {
            "hostname": platform.node(),
            "os": platform.platform(),
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "device": str(config.DEVICE),
            "cuda_available": torch.cuda.is_available(),
            "mps_available": torch.backends.mps.is_available() if hasattr(torch.backends, "mps") else False,
            "cpu_count": torch.get_num_threads() if hasattr(torch, "get_num_threads") else None,
        }
        if torch.cuda.is_available():
            hardware_info["cuda_device"] = torch.cuda.get_device_name(0)
            hardware_info["cuda_device_count"] = torch.cuda.device_count()

        model_info = {
            "predator_policy_params": sum(p.numel() for p in self.predator.policy.parameters()),
            "predator_value_params": sum(p.numel() for p in self.predator.value_net.parameters()),
        }
        if self.is_om:
            model_info["opponent_model_params"] = sum(
                p.numel() for p in self.predator.opponent_model.parameters()
            )

        metadata = {
            "experiment_name": self.experiment_name,
            "is_om": self.is_om,
            "git_commit": git_hash,
            "git_branch": git_branch,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "config": config_snapshot,
            "hardware": hardware_info,
            "model_info": model_info,
        }

        path = self.results_dir / "run_metadata.json"
        with open(path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)
        logger.info(f"Run metadata saved: {path}")

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def _run_eval(self, episode: int) -> dict:
        wins = 0
        total_steps = 0
        total_pred_reward = 0.0

        # Feature 23: Use config.EVAL_DETERMINISTIC to choose stochastic vs deterministic actions
        select_fn = self.predator.select_action_deterministic if config.EVAL_DETERMINISTIC else self.predator.select_action

        for _ in range(config.EVAL_EPISODES):
            (obs_pred, obs_prey), _ = self.env.reset()
            self.predator.reset_episode()
            last_prey_action = 0
            ep_steps = 0

            for _ in range(config.MAX_STEPS):
                ep_steps += 1
                if self.is_om:
                    a_pred = select_fn(
                        obs_pred, last_opp_action=last_prey_action
                    )
                else:
                    a_pred = select_fn(obs_pred)

                a_prey = np.random.randint(0, config.ACTION_DIM)
                (obs_pred, obs_prey), (r_pred, _), term, trunc, info = self.env.step(
                    (a_pred, a_prey)
                )
                last_prey_action = a_prey
                total_pred_reward += r_pred

                if term or trunc:
                    break

            total_steps += ep_steps
            if info["captured"]:
                wins += 1

        win_rate = wins / config.EVAL_EPISODES
        avg_steps = total_steps / config.EVAL_EPISODES
        avg_reward = total_pred_reward / config.EVAL_EPISODES

        # Feature 46: Confidence interval
        if config.EVAL_CONFIDENCE_INTERVAL > 0:
            import math
            z = config.EVAL_CONFIDENCE_INTERVAL  # z-score (e.g., 1.96 for 95%)
            ci = z * math.sqrt((win_rate * (1 - win_rate)) / config.EVAL_EPISODES)
            ci_msg = f" (CI: {win_rate:.2f} ± {ci:.3f})"
        else:
            ci_msg = ""

        logger.info(
            f"Eval @ ep {episode}: win_rate={win_rate:.2f}{ci_msg}, "
            f"avg_steps={avg_steps:.1f}, avg_reward={avg_reward:.3f}"
        )

        result = {
            "episode": episode,
            "win_rate_vs_random": round(win_rate, 4),
            "avg_episode_length": round(avg_steps, 1),
            "avg_predator_reward": round(avg_reward, 4),
        }
        if config.EVAL_CONFIDENCE_INTERVAL > 0:
            z = config.EVAL_CONFIDENCE_INTERVAL
            ci = z * math.sqrt((win_rate * (1 - win_rate)) / config.EVAL_EPISODES)
            result["win_rate_ci"] = round(ci, 4)
        return result

    # ------------------------------------------------------------------
    # Past-Self Evaluation
    # ------------------------------------------------------------------
    def _save_snapshot(self, episode: int) -> None:
        path = self.snapshots_dir / f"snapshot_{episode}.pt"
        state = {
            "predator_policy": copy.deepcopy(self.predator.policy.state_dict()),
            "predator_value": copy.deepcopy(self.predator.value_net.state_dict()),
            "is_om": self.is_om,
        }
        if self.is_om:
            state["predator_om"] = copy.deepcopy(self.predator.opponent_model.state_dict())
        torch.save(state, path)
        logger.info(f"Snapshot saved: {path}")

    def _load_snapshot_agent(self, episode: int) -> BaseAgent:
        path = self.snapshots_dir / f"snapshot_{episode}.pt"
        if not path.exists():
            raise FileNotFoundError(f"Snapshot not found: {path}")

        state = torch.load(path, map_location=config.DEVICE, weights_only=True)

        if self.is_om:
            from agents.om_agent import OMAgent
            agent = OMAgent(obs_dim=config.OBS_DIM, action_dim=config.ACTION_DIM)
            agent.policy.load_state_dict(state["predator_policy"])
            agent.value_net.load_state_dict(state["predator_value"])
            agent.opponent_model.load_state_dict(state["predator_om"])
        else:
            from agents.ppo_agent import PPOAgent
            agent = PPOAgent(obs_dim=config.OBS_DIM, action_dim=config.ACTION_DIM)
            agent.policy.load_state_dict(state["predator_policy"])
            agent.value_net.load_state_dict(state["predator_value"])

        agent.policy.eval()
        agent.value_net.eval()
        if self.is_om:
            agent.opponent_model.eval()

        return agent

    def _run_past_self_eval(self, episode: int) -> float:
        snapshots = sorted(
            [p for p in self.snapshots_dir.glob("snapshot_*.pt")],
            key=lambda p: int(p.stem.split("_")[1]),
        )
        if not snapshots:
            logger.warning("No snapshots available for past-self eval.")
            return 0.0

        latest_snapshot_ep = int(snapshots[-1].stem.split("_")[1])
        try:
            past_agent = self._load_snapshot_agent(latest_snapshot_ep)
        except FileNotFoundError:
            logger.warning(f"Could not load snapshot from ep {latest_snapshot_ep}.")
            return 0.0

        wins = 0

        for _ in range(config.PAST_SELF_EVAL_EPISODES):
            (obs_pred, obs_prey), _ = self.env.reset()
            self.predator.reset_episode()
            past_agent.reset_episode()
            last_pred_action = 0
            last_prey_action = 0

            for _ in range(config.MAX_STEPS):
                if self.is_om:
                    a_pred = self.predator.select_action_deterministic(
                        obs_pred, last_opp_action=last_prey_action
                    )
                    a_prey = past_agent.select_action_deterministic(
                        obs_prey, last_opp_action=last_pred_action
                    )
                else:
                    a_pred = self.predator.select_action_deterministic(obs_pred)
                    a_prey = past_agent.select_action_deterministic(obs_prey)

                (obs_pred, obs_prey), _, term, trunc, info = self.env.step(
                    (a_pred, a_prey)
                )
                last_pred_action, last_prey_action = a_pred, a_prey

                if term or trunc:
                    break

            if info["captured"]:
                wins += 1

        win_rate = wins / config.PAST_SELF_EVAL_EPISODES
        logger.info(
            f"Past-self eval @ ep {episode} (vs snapshot ep {latest_snapshot_ep}): "
            f"win_rate={win_rate:.2f}"
        )

        return win_rate

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------
    def _save_checkpoint(self, tag, metric: float = 0.0) -> None:
        state = {
            "predator_policy": self.predator.policy.state_dict(),
            "predator_value": self.predator.value_net.state_dict(),
            "prey_policy": self.prey.policy.state_dict(),
            "prey_value": self.prey.value_net.state_dict(),
        }
        if self.is_om:
            state["predator_om"] = self.predator.opponent_model.state_dict()
            state["prey_om"] = self.prey.opponent_model.state_dict()

        # --- NEW: Use checkpoint manager ---
        self._ckpt_mgr.save(state, metric, str(tag))
