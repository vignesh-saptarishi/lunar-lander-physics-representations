"""Integration test: train a single agent and verify outputs.

Trains a blind and labeled PPO agent for 10,000 steps (~30 seconds each)
on full-variation easy, then verifies all expected output files exist.
"""

import json
import subprocess
import sys
from pathlib import Path


def run_script(
    args: list[str], cwd: Path, timeout: int = 300
) -> subprocess.CompletedProcess:
    """Run a script as subprocess from the repo root."""
    result = subprocess.run(
        [sys.executable] + args,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.returncode != 0:
        print(f"STDOUT:\n{result.stdout}")
        print(f"STDERR:\n{result.stderr}")
    return result


class TestSingleAgentTraining:
    """Train one agent, verify outputs."""

    def test_train_blind_agent(self, repo_root, tmp_output):
        """Train a blind agent for 10K steps and check output files."""
        run_dir = tmp_output / "blind-test-s42"

        result = run_script(
            [
                "lunar_lander/scripts/train_rl.py",
                "--variant",
                "blind",
                "--algo",
                "ppo",
                "--profile",
                "easy",
                "--total-steps",
                "20000",
                "--n-envs",
                "2",
                "--seed",
                "42",
                "--run-dir",
                str(run_dir),
                "--video-freq",
                "0",
                "--eval-freq",
                "20000",
                "--checkpoint-freq",
                "20000",
            ],
            cwd=repo_root,
        )

        assert result.returncode == 0, f"Training failed:\n{result.stderr}"

        # Core outputs exist
        assert (run_dir / "model.zip").exists(), "Final model not saved"
        assert (run_dir / "config.json").exists(), "Config not saved"
        assert (run_dir / "vec_normalize.pkl").exists(), "VecNormalize stats not saved"

        # Config is valid JSON with expected fields
        config = json.loads((run_dir / "config.json").read_text())
        assert config["variant"] == "blind"
        assert config["algo"] == "ppo"
        assert config["seed"] == 42

    def test_train_labeled_agent(self, repo_root, tmp_output):
        """Train a labeled agent for 10K steps and check output files."""
        run_dir = tmp_output / "labeled-test-s42"

        result = run_script(
            [
                "lunar_lander/scripts/train_rl.py",
                "--variant",
                "labeled",
                "--algo",
                "ppo",
                "--profile",
                "easy",
                "--total-steps",
                "20000",
                "--n-envs",
                "2",
                "--seed",
                "42",
                "--run-dir",
                str(run_dir),
                "--video-freq",
                "0",
                "--eval-freq",
                "20000",
                "--checkpoint-freq",
                "20000",
            ],
            cwd=repo_root,
        )

        assert result.returncode == 0, f"Training failed:\n{result.stderr}"
        assert (run_dir / "model.zip").exists()
        assert (run_dir / "config.json").exists()

        config = json.loads((run_dir / "config.json").read_text())
        assert config["variant"] == "labeled"
