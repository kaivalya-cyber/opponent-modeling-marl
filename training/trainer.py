"""
Training loop for self-play multi-agent RL.

1. Initialize env, agent_predator, agent_prey (same class for self-play)
2. Curriculum learning: progressive grid sizes (5→7→10)
3. Collect ROLLOUT_STEPS transitions, then run PPO update for both agents
4. Log metrics to CSV every episode + wandb (optional)
5. Run evaluation against random opponent every EVAL_INTERVAL episodes
6. Run past-self evaluation (pit current agent vs historical snapshot)
7. Save checkpoints every CHECKPOINT_INTERVAL episodes
"""

from __future__ import annotations

import copy
import csv
import json
import logging
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from collections import deque

import numpy as np
import torch

import config
from env.predator_prey import PredatorPreyEnv
from agents.base_agent import BaseAgent
from training.rollout_buffer import RolloutBuffer
from training.elo import EloRating

logger = logging.getLogger(__name__)

# Optional wandb import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


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

        # Determine obs dim for buffers (OM agents store augmented 34-dim obs)
        self.buf_obs_dim = config.OBS_DIM + config.ACTION_DIM if is_om else config.OBS_DIM

        self.elo = EloRating()

        # Results directory
        self.results_dir = Path("results") / experiment_name
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Snapshots directory for past-self evaluation
        self.snapshots_dir = self.results_dir / "snapshots"
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)

        # Wandb initialization
        self._use_wandb = config.USE_WANDB and WANDB_AVAILABLE
        if self._use_wandb:
            wandb.init(
                project=config.WANDB_PROJECT,
                name=experiment_name,
                config={
                    "is_om": is_om,
                    "grid_size": config.GRID_SIZE,
                    "total_episodes": total_episodes,
                    "learning_rate": config.LEARNING_RATE,
                    "curriculum_enabled": config.CURRICULUM_ENABLED,
                },
            )
        elif config.USE_WANDB and not WANDB_AVAILABLE:
            logger.warning("wandb not installed — set USE_WANDB=False or pip install wandb")

        # Curriculum learning tracking
        self._curriculum_enabled = config.CURRICULUM_ENABLED
        self._curriculum_stage = 0  # index into config.CURRICULUM_SCHEDULE

        # CSV logging
        self.csv_path = self.results_dir / "metrics.csv"
        self.eval_csv_path = self.results_dir / "eval_metrics.csv"
        self._init_csv()
        self._init_eval_csv()

        # Tracking
        self.recent_captures = deque(maxlen=20)

        # Save run metadata for reproducibility
        self._save_run_metadata()

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
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

    def _init_eval_csv(self) -> None:
        """Initialize the evaluation metrics CSV."""
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
        """Return (grid_size, max_steps) for the current curriculum stage.

        Walks the CURRICULUM_SCHEDULE and advances the stage when
        episode passes the next threshold.
        """
        if not self._curriculum_enabled:
            return config.GRID_SIZE, config.MAX_STEPS

        schedule = config.CURRICULUM_SCHEDULE
        # Advance to the latest stage that starts at or before this episode
        while (
            self._curriculum_stage + 1 < len(schedule)
            and episode >= schedule[self._curriculum_stage + 1][0]
        ):
            self._curriculum_stage += 1

        _, grid_size, max_steps = schedule[self._curriculum_stage]
        return grid_size, max_steps

    def _ensure_env_matches_curriculum(self, episode: int) -> bool:
        """Recreate the environment if the curriculum stage changed.

        Returns True if the environment was recreated.
        """
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

        global_step = 0
        episode = 0
        pred_metrics: dict = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
        if self.is_om:
            pred_metrics["om_loss"] = 0.0

        while episode < self.total_episodes:
            # Curriculum: adjust environment if stage changed
            if self._ensure_env_matches_curriculum(episode):
                # If curriculum changed grid size and past-self eval is active,
                # note that past snapshots were trained on smaller grids —
                # past-self win rates may be inflated at larger grid sizes.
                if config.PAST_SELF_EVAL and episode >= config.SNAPSHOT_INTERVAL:
                    logger.info(
                        "Curriculum grid changed while past-self eval is active. "
                        "Past snapshots were trained on smaller grids — "
                        "past-self win rates may be inflated."
                    )

            (obs_pred, obs_prey), _ = self.env.reset()
            self.predator.reset_episode()
            self.prey.reset_episode()

            last_pred_action = 0
            last_prey_action = 0
            ep_reward_pred = 0.0
            ep_reward_prey = 0.0
            done = False

            while not done:
                # Select actions
                if self.is_om:
                    aug_pred = self.predator.get_augmented_obs(obs_pred, last_prey_action)
                    a_pred, lp_pred, v_pred = self.predator.select_action(
                        obs_pred, last_opp_action=last_prey_action
                    )
                    aug_prey = self.prey.get_augmented_obs(obs_prey, last_pred_action)
                    a_prey, lp_prey, v_prey = self.prey.select_action(
                        obs_prey, last_opp_action=last_pred_action
                    )
                else:
                    a_pred, lp_pred, v_pred = self.predator.select_action(obs_pred)
                    a_prey, lp_prey, v_prey = self.prey.select_action(obs_prey)

                # Step environment
                (next_obs_pred, next_obs_prey), (r_pred, r_prey), terminated, truncated, info = (
                    self.env.step((a_pred, a_prey))
                )
                done = terminated or truncated

                # Store transitions
                store_obs_pred = aug_pred.numpy() if self.is_om else obs_pred
                store_obs_prey = aug_prey.numpy() if self.is_om else obs_prey

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

                # Update when both buffers are full
                if pred_buf.full and prey_buf.full:
                    # Get last values for GAE
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

                    pred_buf.reset()
                    prey_buf.reset()

            # Episode finished
            captured = info["captured"]
            self.recent_captures.append(1.0 if captured else 0.0)

            # Update Elo
            if captured:
                self.elo.update("predator", "prey")
            else:
                self.elo.draw("predator", "prey")

            capture_rate = sum(self.recent_captures) / len(self.recent_captures)

            # Decay OM loss weight (shift from opponent modeling → pure RL)
            if self.is_om:
                self.predator.om_loss_weight *= config.OM_LOSS_WEIGHT_DECAY
                self.prey.om_loss_weight *= config.OM_LOSS_WEIGHT_DECAY

            # Prepare loss metrics (use predator metrics as representative)
            pol_loss = round(pred_metrics["policy_loss"], 6)
            val_loss = round(pred_metrics["value_loss"], 6)
            ent = round(pred_metrics["entropy"], 6)

            # Log training row
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
            if self._curriculum_enabled:
                row["grid_size"] = self.env.grid_size
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
                if self._curriculum_enabled:
                    wandb_log["grid_size"] = self.env.grid_size
                wandb.log(wandb_log, step=episode)

            if episode % 50 == 0:
                log_msg = (
                    f"Ep {episode:>4d} | Cap rate: {capture_rate:.2f} | "
                    f"Pred Elo: {self.elo.get_rating('predator'):.0f} | "
                    f"Prey Elo: {self.elo.get_rating('prey'):.0f}"
                )
                if self.is_om:
                    log_msg += f" | OM w: {self.predator.om_loss_weight:.4f}"
                if self._curriculum_enabled:
                    log_msg += f" | grid: {self.env.grid_size}"
                logger.info(log_msg)

            # Evaluation against random opponent (deterministic actions)
            if episode > 0 and episode % config.EVAL_INTERVAL == 0:
                eval_row = self._run_eval(episode)

                # Wandb eval logging
                if self._use_wandb:
                    wandb.log({
                        "eval/win_rate_vs_random": eval_row["win_rate_vs_random"],
                        "eval/avg_episode_length": eval_row["avg_episode_length"],
                        "eval/avg_predator_reward": eval_row["avg_predator_reward"],
                    }, step=episode)

                # Past-self evaluation
                if config.PAST_SELF_EVAL and episode >= config.SNAPSHOT_INTERVAL:
                    past_win_rate = self._run_past_self_eval(episode)
                    eval_row["win_rate_vs_past_self"] = round(past_win_rate, 4)
                    if self._use_wandb:
                        wandb.log(
                            {"eval/win_rate_vs_past_self": past_win_rate},
                            step=episode,
                        )

                self._log_eval_row(eval_row)

            # Save snapshot for past-self evaluation
            if (
                episode > 0
                and episode % config.SNAPSHOT_INTERVAL == 0
                and config.PAST_SELF_EVAL
            ):
                self._save_snapshot(episode)

            # Checkpoint
            if episode > 0 and episode % config.CHECKPOINT_INTERVAL == 0:
                self._save_checkpoint(episode)

            episode += 1

        # Final checkpoint
        self._save_checkpoint("final")

        if self._use_wandb:
            wandb.finish()

        logger.info("Training complete.")

    # ------------------------------------------------------------------
    # Run Metadata
    # ------------------------------------------------------------------
    def _save_run_metadata(self) -> None:
        """Save git hash, config snapshot, and hardware info for reproducibility.

        Creates a run_metadata.json file in the results directory containing:
        - Git commit hash and branch
        - Full config values (except device-specific settings)
        - Hardware info (hostname, OS, Python version, PyTorch version, device)
        - Training start timestamp
        - Experiment name and type (baseline vs OM)
        """
        # Git info
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

        # Config snapshot — capture all uppercase config variables
        config_snapshot = {}
        for key in dir(config):
            if key.isupper() and not key.startswith("_"):
                value = getattr(config, key)
                # Make serializable
                if isinstance(value, (str, int, float, bool, type(None))):
                    config_snapshot[key] = value
                elif isinstance(value, (list, tuple)):
                    config_snapshot[key] = list(value)
                else:
                    config_snapshot[key] = str(value)

        # Hardware info
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

        # Model parameter counts
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
        """Run deterministic evaluation of predator against random prey.

        Uses greedy (argmax) action selection — no sampling noise —
        to assess the true quality of the learned policy.

        Returns dict with eval metrics for CSV logging.
        """
        wins = 0
        total_steps = 0
        total_pred_reward = 0.0

        for _ in range(config.EVAL_EPISODES):
            (obs_pred, obs_prey), _ = self.env.reset()
            self.predator.reset_episode()
            last_prey_action = 0
            ep_steps = 0

            for _ in range(config.MAX_STEPS):
                ep_steps += 1
                if self.is_om:
                    a_pred = self.predator.select_action_deterministic(
                        obs_pred, last_opp_action=last_prey_action
                    )
                else:
                    a_pred = self.predator.select_action_deterministic(obs_pred)

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

        logger.info(
            f"Eval @ ep {episode}: win_rate={win_rate:.2f}, "
            f"avg_steps={avg_steps:.1f}, avg_reward={avg_reward:.3f}"
        )

        return {
            "episode": episode,
            "win_rate_vs_random": round(win_rate, 4),
            "avg_episode_length": round(avg_steps, 1),
            "avg_predator_reward": round(avg_reward, 4),
        }

    # ------------------------------------------------------------------
    # Past-Self Evaluation
    # ------------------------------------------------------------------
    def _save_snapshot(self, episode: int) -> None:
        """Save a deep copy of the predator's weights for past-self eval."""
        # Save only the predator (evaluate predator vs its past self)
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
        """Load a past snapshot and return a frozen evaluation agent.

        The returned agent has its weights loaded from the snapshot and
        is set to eval mode (no gradients).
        """
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
        """Pit current predator against the most recent past snapshot.

        The past snapshot is loaded, frozen (eval mode), and used as the
        prey. Both agents use deterministic action selection. This detects
        whether the agent is genuinely improving or just cycling strategies.

        Returns:
            win_rate: fraction of episodes the current agent wins (as predator).
        """
        # Find the most recent snapshot at or before this episode
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
                # Current agent is predator, past snapshot is prey
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
    def _save_checkpoint(self, tag) -> None:
        """Save model checkpoints."""
        path = self.results_dir / f"checkpoint_{tag}.pt"
        state = {
            "predator_policy": self.predator.policy.state_dict(),
            "predator_value": self.predator.value_net.state_dict(),
            "prey_policy": self.prey.policy.state_dict(),
            "prey_value": self.prey.value_net.state_dict(),
        }
        if self.is_om:
            state["predator_om"] = self.predator.opponent_model.state_dict()
            state["prey_om"] = self.prey.opponent_model.state_dict()
        torch.save(state, path)
        logger.info(f"Checkpoint saved: {path}")
