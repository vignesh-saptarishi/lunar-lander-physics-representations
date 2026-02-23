"""Integration test: run eval pipeline on a trained agent.

Trains a short agent, then runs collect_trajectories + compute_metrics.
Verifies .npz trajectory files and metrics.csv are produced with correct structure.
"""

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest


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


@pytest.fixture
def trained_agent(repo_root, tmp_output):
    """Train a minimal agent and return its run_dir."""
    run_dir = tmp_output / "eval-test-agent"

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
    return run_dir


class TestEvalPipeline:
    """Collect trajectories, compute metrics, verify outputs."""

    def test_collect_trajectories(self, repo_root, trained_agent):
        """Collect 10 episodes and verify .npz files exist."""
        traj_dir = trained_agent / "trajectories-test"

        result = run_script(
            [
                "lunar_lander/scripts/collect_trajectories.py",
                "--checkpoint-dir",
                str(trained_agent),
                "--episodes",
                "10",
                "--seed",
                "0",
                "--output-dir",
                str(traj_dir),
            ],
            cwd=repo_root,
        )

        assert result.returncode == 0, f"Collection failed:\n{result.stderr}"

        npz_files = list(traj_dir.glob("episode_*.npz"))
        assert len(npz_files) == 10, f"Expected 10 .npz files, got {len(npz_files)}"

    def test_compute_metrics(self, repo_root, trained_agent):
        """Collect trajectories then compute metrics. Verify metrics.csv structure."""
        traj_dir = trained_agent / "trajectories-metrics-test"

        # Collect
        result = run_script(
            [
                "lunar_lander/scripts/collect_trajectories.py",
                "--checkpoint-dir",
                str(trained_agent),
                "--episodes",
                "10",
                "--seed",
                "0",
                "--output-dir",
                str(traj_dir),
            ],
            cwd=repo_root,
        )
        assert result.returncode == 0

        # Compute metrics
        result = run_script(
            [
                "lunar_lander/scripts/compute_metrics.py",
                str(traj_dir),
            ],
            cwd=repo_root,
        )
        assert result.returncode == 0, f"Metrics failed:\n{result.stderr}"

        # Verify metrics.csv
        metrics_csv = traj_dir / "metrics.csv"
        assert metrics_csv.exists(), "metrics.csv not created"

        df = pd.read_csv(metrics_csv)
        assert len(df) == 10, f"Expected 10 rows, got {len(df)}"

        # Required columns exist
        required_cols = [
            "outcome",
            "total_reward",
            "episode_steps",
            "thrust_duty_cycle",
            "total_fuel",
            "gravity",
            "main_engine_power",
            "twr",
        ]
        for col in required_cols:
            assert col in df.columns, f"Missing column: {col}"

        # Outcomes are valid
        valid_outcomes = {"landed", "crashed", "timeout", "out_of_bounds"}
        assert set(df["outcome"].unique()).issubset(valid_outcomes)

    def test_run_eval_pipeline(self, repo_root, trained_agent):
        """Run the full eval pipeline script and verify behavioral_summary.json."""
        result = run_script(
            [
                "lunar_lander/scripts/run_eval_pipeline.py",
                "--checkpoint-dir",
                str(trained_agent),
                "--episodes",
                "10",
                "--trajectory-subdir",
                "trajectories-pipeline-test",
            ],
            cwd=repo_root,
        )
        assert result.returncode == 0, f"Pipeline failed:\n{result.stderr}"

        traj_dir = trained_agent / "trajectories-pipeline-test"
        assert (traj_dir / "metrics.csv").exists()

        # Behavioral analysis outputs
        analysis_dir = traj_dir / "behavioral_analysis"
        assert analysis_dir.exists(), "behavioral_analysis/ not created"
        assert (analysis_dir / "behavioral_summary.json").exists()
