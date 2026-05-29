import sys
import os
import shutil
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(__file__))


# ---------------------------------------------------------------------------
# Fixture: clean up results/test_* directories after integration tests
# ---------------------------------------------------------------------------

@pytest.fixture
def _cleanup_test_results_dirs():
    """Clean up results/test_* directories left by integration tests.

    Integration tests create Trainer instances which generate directories
    like results/test_no_wandb, results/test_snapshot, etc. Use this fixture
    with usefixtures on integration test classes so it only runs where needed.
    """
    yield  # Run the test

    results_dir = Path(__file__).parent / "results"
    if results_dir.exists():
        for subdir in results_dir.iterdir():
            if subdir.is_dir() and subdir.name.startswith("test_"):
                try:
                    shutil.rmtree(subdir)
                except (FileNotFoundError, PermissionError):
                    pass  # Already gone or permission issue — skip
