"""Integration test: multi-seed training + cross-config comparison.

Trains 2 blind + 2 labeled agents (10K steps each), runs eval pipeline,
writes a comparison manifest, and runs compare_configs.py.
Verifies stat_tests.json is produced with valid structure.

This is the critical test: it proves the full Article 1 reproduction
pipeline works end-to-end.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml


def run_script(
    args: list[str], cwd: Path, timeout: int = 600
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


def train_agent(repo_root: Path, run_dir: Path, variant: str, seed: int):
    """Train one agent with minimal steps."""
    result = run_script(
        [
            "lunar_lander/scripts/train_rl.py",
            "--variant",
            variant,
            "--algo",
            "ppo",
            "--profile",
            "easy",
            "--total-steps",
            "20000",
            "--n-envs",
            "2",
            "--seed",
            str(seed),
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
    assert (
        result.returncode == 0
    ), f"Training {variant} s{seed} failed:\n{result.stderr}"


def eval_agent(repo_root: Path, checkpoint_dir: Path, episodes: int = 20):
    """Run eval pipeline on one agent."""
    result = run_script(
        [
            "lunar_lander/scripts/run_eval_pipeline.py",
            "--checkpoint-dir",
            str(checkpoint_dir),
            "--episodes",
            str(episodes),
            "--skip-analysis",
        ],
        cwd=repo_root,
    )
    assert (
        result.returncode == 0
    ), f"Eval {checkpoint_dir.name} failed:\n{result.stderr}"


class TestMultiSeedComparison:
    """Train multiple seeds, compare, verify statistical output."""

    @pytest.fixture(autouse=True)
    def setup_agents(self, repo_root, tmp_output):
        """Train 2 blind + 2 labeled agents and run eval on each."""
        self.repo_root = repo_root
        self.output_dir = tmp_output / "multi-seed-test"
        self.labeled_base = self.output_dir / "labeled"
        self.blind_base = self.output_dir / "blind"

        seeds = [42, 123]

        # Train all 4 agents
        for seed in seeds:
            train_agent(repo_root, self.labeled_base / f"s{seed}", "labeled", seed)
            train_agent(repo_root, self.blind_base / f"s{seed}", "blind", seed)

        # Eval all 4 agents
        for seed in seeds:
            eval_agent(repo_root, self.labeled_base / f"s{seed}")
            eval_agent(repo_root, self.blind_base / f"s{seed}")

    def test_comparison_produces_stat_tests(self, tmp_output):
        """Run compare_configs.py and verify stat_tests.json output."""
        comparison_output = self.output_dir / "comparison-results"

        # Write manifest with actual paths
        manifest_path = self.output_dir / "test-manifest.yaml"
        manifest = {
            "experiment": "test-comparison",
            "output_base": str(comparison_output),
            "comparisons": {
                "test-condition": {
                    "condition": "full-variation",
                    "profile": "easy",
                    "configs": {
                        "labeled": {
                            "seed_base": str(self.labeled_base),
                            "seeds": [42, 123],
                            "trajectory_subdir": "trajectories",
                        },
                        "blind": {
                            "seed_base": str(self.blind_base),
                            "seeds": [42, 123],
                            "trajectory_subdir": "trajectories",
                        },
                    },
                },
            },
        }
        manifest_path.write_text(yaml.dump(manifest))

        # Run comparison
        result = run_script(
            [
                "lunar_lander/scripts/compare_configs.py",
                "--manifest",
                str(manifest_path),
            ],
            cwd=self.repo_root,
        )
        assert result.returncode == 0, f"Comparison failed:\n{result.stderr}"

        # Verify stat_tests.json exists and has valid structure
        stat_tests = comparison_output / "stat_tests.json"
        assert stat_tests.exists(), "stat_tests.json not created"

        data = json.loads(stat_tests.read_text())

        # Top-level keys are comparison names
        assert "test-condition" in data

        comparison = data["test-condition"]

        # landed_pct metric exists with expected structure
        assert "landed_pct" in comparison
        landed = comparison["landed_pct"]
        assert "p_value" in landed
        assert "effect_size_cohens_d" in landed
        assert "per_variant" in landed

        # Per-variant stats for both labeled and blind
        for variant in ["labeled", "blind"]:
            assert variant in landed["per_variant"]
            stats = landed["per_variant"][variant]
            assert "mean" in stats
            assert "per_seed" in stats
            assert len(stats["per_seed"]) == 2  # 2 seeds
