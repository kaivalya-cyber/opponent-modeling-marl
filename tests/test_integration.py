"""
Integration tests for high-level systems: tournament, wandb logging,
curriculum learning, and past-self evaluation.

These tests verify that the system components work together correctly.
"""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, patch
from pathlib import Path

import numpy as np
import pytest
import torch

import config


# ---------------------------------------------------------------------------
# Tournament integration tests
# ---------------------------------------------------------------------------

class TestTournamentLogic:
    """Test the tournament match-running logic without actual checkpoints."""

    def test_run_match_basic_stats(self):
        """run_match should return correct dict keys and bounded values."""
        from experiments.tournament import run_match
        from agents.ppo_agent import PPOAgent
        from env.predator_prey import PredatorPreyEnv

        env = PredatorPreyEnv(grid_size=5, max_steps=50)
        predator = PPOAgent(obs_dim=29, action_dim=5)
        prey = PPOAgent(obs_dim=29, action_dim=5)

        stats = run_match(env, predator, prey, False, False, n_matches=10)

        assert set(stats.keys()) == {
            "wins_predator",
            "wins_prey",
            "draws",
            "avg_steps",
            "avg_pred_reward",
            "avg_prey_reward",
        }
        assert 0 <= stats["wins_predator"] <= 10
        assert 0 <= stats["wins_prey"] <= 10
        assert stats["wins_predator"] + stats["wins_prey"] == 10
        assert stats["draws"] == 0  # no draw logic in current implementation
        assert stats["avg_steps"] > 0
        # Predator gets -0.1 per step, so reward can be negative
        assert -10.0 <= stats["avg_pred_reward"] <= 10.0
        assert -10.0 <= stats["avg_prey_reward"] <= 10.0

    def test_run_match_om_vs_ppo(self):
        """OM agent should work as predator in tournament match."""
        from experiments.tournament import run_match
        from agents.ppo_agent import PPOAgent
        from agents.om_agent import OMAgent
        from env.predator_prey import PredatorPreyEnv

        env = PredatorPreyEnv(grid_size=5, max_steps=50)
        om = OMAgent(obs_dim=29, action_dim=5)
        ppo = PPOAgent(obs_dim=29, action_dim=5)

        om.reset_episode()
        ppo.reset_episode()

        stats = run_match(env, om, ppo, True, False, n_matches=10)

        assert stats["wins_predator"] + stats["wins_prey"] == 10
        assert stats["avg_steps"] > 0

    def test_tournament_main_imports(self):
        """Tournament main function should be importable."""
        from experiments.tournament import main
        assert callable(main)

    def test_load_agent_file_not_found(self):
        """load_agent should raise FileNotFoundError for missing checkpoints."""
        from experiments.tournament import load_agent

        with pytest.raises(FileNotFoundError):
            load_agent(Path("/nonexistent/checkpoint.pt"), is_om=False)


# ---------------------------------------------------------------------------
# Wandb integration tests (mocked)
# ---------------------------------------------------------------------------

class TestWandbIntegration:
    """Test that wandb integration doesn't crash and handles missing wandb."""

    @pytest.fixture(autouse=True)
    def _cleanup(self):
        """Clean up results/test_* directories after each test in this class."""
        yield
        from pathlib import Path
        import shutil
        results_dir = Path(__file__).parent.parent / "results"
        if results_dir.exists():
            for subdir in results_dir.iterdir():
                if subdir.is_dir() and subdir.name.startswith("test_"):
                    try:
                        shutil.rmtree(subdir)
                    except (FileNotFoundError, PermissionError):
                        pass

    def setup_method(self):
        """Ensure wandb is not imported from any previous test."""
        # Remove wandb from sys.modules if present
        sys.modules.pop("wandb", None)

    def test_trainer_without_wandb_installed(self):
        """Trainer should work when wandb is not installed."""
        from agents.ppo_agent import PPOAgent
        from training.trainer import Trainer

        original_use_wandb = config.USE_WANDB
        try:
            config.USE_WANDB = False

            predator = PPOAgent(obs_dim=29, action_dim=5)
            prey = PPOAgent(obs_dim=29, action_dim=5)

            trainer = Trainer(predator, prey, "test_no_wandb", total_episodes=1)
            assert not trainer._use_wandb
        finally:
            config.USE_WANDB = original_use_wandb

    @patch.dict("sys.modules", {"wandb": MagicMock()})
    def test_trainer_with_mocked_wandb(self):
        """Trainer should initialize wandb when USE_WANDB=True and wandb available."""
        import wandb as mock_wandb

        original_use_wandb = config.USE_WANDB
        config.USE_WANDB = True

        # Save the original WANDB_AVAILABLE before reloading with mock
        import training.trainer
        _original_wandb_available = training.trainer.WANDB_AVAILABLE

        try:
            import importlib
            importlib.reload(training.trainer)

            from agents.ppo_agent import PPOAgent
            from training.trainer import Trainer

            predator = PPOAgent(obs_dim=29, action_dim=5)
            prey = PPOAgent(obs_dim=29, action_dim=5)

            trainer = Trainer(predator, prey, "test_wandb", total_episodes=1)
            assert trainer._use_wandb
            mock_wandb.init.assert_called_once()
        finally:
            config.USE_WANDB = original_use_wandb
            # Restore WANDB_AVAILABLE directly (reload would still pick up the mock)
            training.trainer.WANDB_AVAILABLE = _original_wandb_available

    def test_trainer_has_logging_methods(self):
        """Trainer should have CSV logging methods for training and eval."""
        from agents.ppo_agent import PPOAgent
        from training.trainer import Trainer

        original_use_wandb = config.USE_WANDB
        try:
            config.USE_WANDB = False

            predator = PPOAgent(obs_dim=29, action_dim=5)
            prey = PPOAgent(obs_dim=29, action_dim=5)
            trainer = Trainer(predator, prey, "test_log", total_episodes=5)

            # CSV logging methods should exist
            assert hasattr(trainer, "_use_wandb")
            assert hasattr(trainer, "_log_row")
            assert hasattr(trainer, "_log_eval_row")
        finally:
            config.USE_WANDB = original_use_wandb


# ---------------------------------------------------------------------------
# Curriculum learning integration tests
# ---------------------------------------------------------------------------

class TestCurriculumLearning:
    """Test curriculum learning stage transitions and env recreation."""

    @pytest.fixture(autouse=True)
    def _cleanup(self):
        """Clean up results/test_* directories after each test in this class."""
        yield
        from pathlib import Path
        import shutil
        results_dir = Path(__file__).parent.parent / "results"
        if results_dir.exists():
            for subdir in results_dir.iterdir():
                if subdir.is_dir() and subdir.name.startswith("test_"):
                    try:
                        shutil.rmtree(subdir)
                    except (FileNotFoundError, PermissionError):
                        pass

    def test_get_curriculum_params_disabled(self):
        """When curriculum is disabled, always return default params."""
        from agents.ppo_agent import PPOAgent
        from training.trainer import Trainer

        original_curriculum = config.CURRICULUM_ENABLED
        config.CURRICULUM_ENABLED = False

        try:
            predator = PPOAgent(obs_dim=29, action_dim=5)
            prey = PPOAgent(obs_dim=29, action_dim=5)
            trainer = Trainer(predator, prey, "test_curr", total_episodes=100)

            grid, steps = trainer._get_curriculum_params(0)
            assert grid == config.GRID_SIZE
            assert steps == config.MAX_STEPS

            grid, steps = trainer._get_curriculum_params(600)
            assert grid == config.GRID_SIZE
            assert steps == config.MAX_STEPS
        finally:
            config.CURRICULUM_ENABLED = original_curriculum

    def test_get_curriculum_params_enabled(self):
        """When curriculum is enabled, stage should change at thresholds."""
        from agents.ppo_agent import PPOAgent
        from training.trainer import Trainer

        original_curriculum = config.CURRICULUM_ENABLED
        config.CURRICULUM_ENABLED = True

        try:
            predator = PPOAgent(obs_dim=29, action_dim=5)
            prey = PPOAgent(obs_dim=29, action_dim=5)
            trainer = Trainer(predator, prey, "test_curr", total_episodes=2000)

            # Stage 0: (0, 5, 100)
            grid, steps = trainer._get_curriculum_params(0)
            assert grid == 5
            assert steps == 100

            # Still stage 0
            grid, steps = trainer._get_curriculum_params(499)
            assert grid == 5

            # Stage 1: (500, 7, 150)
            grid, steps = trainer._get_curriculum_params(500)
            assert grid == 7
            assert steps == 150

            # Stage 2: (1000, 10, 200)
            grid, steps = trainer._get_curriculum_params(1000)
            assert grid == 10
            assert steps == 200

            # Beyond last stage — stays at stage 2
            grid, steps = trainer._get_curriculum_params(3000)
            assert grid == 10
            assert steps == 200
        finally:
            config.CURRICULUM_ENABLED = original_curriculum

    def test_ensure_env_recreated_on_stage_change(self):
        """_ensure_env_matches_curriculum should recreate env when grid size changes."""
        from agents.ppo_agent import PPOAgent
        from training.trainer import Trainer

        original_curriculum = config.CURRICULUM_ENABLED
        config.CURRICULUM_ENABLED = True

        try:
            predator = PPOAgent(obs_dim=29, action_dim=5)
            prey = PPOAgent(obs_dim=29, action_dim=5)
            trainer = Trainer(predator, prey, "test_curr", total_episodes=2000)

            original_env = trainer.env
            # Simulate episode 500 (stage 1: grid=7)
            trainer._curriculum_stage = 0
            trainer.env.grid_size = 5
            recreated = trainer._ensure_env_matches_curriculum(500)
            assert recreated
            assert trainer.env.grid_size == 7
            assert trainer.env is not original_env
        finally:
            config.CURRICULUM_ENABLED = original_curriculum

    def test_curriculum_csv_has_grid_size_field(self):
        """When curriculum is enabled, CSV row should include grid_size."""
        from agents.ppo_agent import PPOAgent
        from training.trainer import Trainer

        original_curriculum = config.CURRICULUM_ENABLED
        config.CURRICULUM_ENABLED = True

        try:
            predator = PPOAgent(obs_dim=29, action_dim=5)
            prey = PPOAgent(obs_dim=29, action_dim=5)
            trainer = Trainer(predator, prey, "test_curr", total_episodes=1)

            assert trainer._curriculum_enabled
            # CSV header should contain grid_size
            with open(trainer.csv_path) as f:
                header = f.readline().strip()
                assert "grid_size" in header
        finally:
            config.CURRICULUM_ENABLED = original_curriculum


# ---------------------------------------------------------------------------
# Past-self evaluation integration tests
# ---------------------------------------------------------------------------

class TestPastSelfEval:
    """Test past-self snapshot saving and loading."""

    @pytest.fixture(autouse=True)
    def _cleanup(self):
        """Clean up results/test_* directories after each test in this class."""
        yield
        from pathlib import Path
        import shutil
        results_dir = Path(__file__).parent.parent / "results"
        if results_dir.exists():
            for subdir in results_dir.iterdir():
                if subdir.is_dir() and subdir.name.startswith("test_"):
                    try:
                        shutil.rmtree(subdir)
                    except (FileNotFoundError, PermissionError):
                        pass

    def test_save_and_load_snapshot_baseline(self):
        """Save a snapshot and verify it can be loaded."""
        from agents.ppo_agent import PPOAgent
        from training.trainer import Trainer

        original_past_self = config.PAST_SELF_EVAL
        config.PAST_SELF_EVAL = True

        try:
            predator = PPOAgent(obs_dim=29, action_dim=5)
            prey = PPOAgent(obs_dim=29, action_dim=5)
            trainer = Trainer(predator, prey, "test_snapshot", total_episodes=100)

            # Save a snapshot at episode 100
            trainer._save_snapshot(100)

            # Verify snapshot exists
            snapshot_path = trainer.snapshots_dir / "snapshot_100.pt"
            assert snapshot_path.exists()

            # Load it back
            loaded = trainer._load_snapshot_agent(100)
            assert loaded is not None
            assert isinstance(loaded, PPOAgent)

            # Verify weights match
            for p_orig, p_loaded in zip(
                predator.policy.parameters(), loaded.policy.parameters()
            ):
                assert torch.allclose(p_orig, p_loaded)

            # Clean up
            snapshot_path.unlink()
            trainer.snapshots_dir.rmdir()
        finally:
            config.PAST_SELF_EVAL = original_past_self

    def test_save_and_load_snapshot_om(self):
        """Save and load a snapshot with OM agent (includes opponent model)."""
        from agents.om_agent import OMAgent
        from training.trainer import Trainer

        original_past_self = config.PAST_SELF_EVAL
        config.PAST_SELF_EVAL = True

        try:
            predator = OMAgent(obs_dim=29, action_dim=5)
            prey = OMAgent(obs_dim=29, action_dim=5)
            trainer = Trainer(
                predator, prey, "test_snapshot_om", total_episodes=100, is_om=True,
            )

            trainer._save_snapshot(100)

            snapshot_path = trainer.snapshots_dir / "snapshot_100.pt"
            assert snapshot_path.exists()

            loaded = trainer._load_snapshot_agent(100)
            assert isinstance(loaded, OMAgent)

            # Verify opponent model weights match
            for p_orig, p_loaded in zip(
                predator.opponent_model.parameters(),
                loaded.opponent_model.parameters(),
            ):
                assert torch.allclose(p_orig, p_loaded)

            # Clean up
            snapshot_path.unlink()
            trainer.snapshots_dir.rmdir()
        finally:
            config.PAST_SELF_EVAL = original_past_self

    def test_load_nonexistent_snapshot_raises(self):
        """Loading a snapshot that doesn't exist should raise FileNotFoundError."""
        from agents.ppo_agent import PPOAgent
        from training.trainer import Trainer

        original_past_self = config.PAST_SELF_EVAL
        config.PAST_SELF_EVAL = True

        try:
            predator = PPOAgent(obs_dim=29, action_dim=5)
            prey = PPOAgent(obs_dim=29, action_dim=5)
            trainer = Trainer(predator, prey, "test_snapshot", total_episodes=100)

            with pytest.raises(FileNotFoundError):
                trainer._load_snapshot_agent(9999)
        finally:
            config.PAST_SELF_EVAL = original_past_self

    def test_run_past_self_eval_no_snapshots(self):
        """Should return 0.0 when no snapshots exist."""
        from agents.ppo_agent import PPOAgent
        from training.trainer import Trainer

        original_past_self = config.PAST_SELF_EVAL
        config.PAST_SELF_EVAL = True

        try:
            predator = PPOAgent(obs_dim=29, action_dim=5)
            prey = PPOAgent(obs_dim=29, action_dim=5)
            trainer = Trainer(predator, prey, "test_snapshot", total_episodes=100)

            win_rate = trainer._run_past_self_eval(500)
            assert win_rate == 0.0
        finally:
            config.PAST_SELF_EVAL = original_past_self
