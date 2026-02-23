# lunar-lander-physics-representations

Reproduction code for *"What Do RL Agents Actually Learn About Their World?"* — a study of how observation design shapes what reinforcement learning agents learn about environment physics.

We train PPO agents on a parameterized Lunar Lander environment where 7 physics parameters vary across episodes. Agents that receive physics labels in their observations ("labeled") are compared against agents that must infer physics from dynamics alone ("blind"). The results challenge the assumption that more information leads to better performance.

## Quick Start

```bash
# Pipeline test (~5 min, 2 seeds, 32K steps)
./reproduce.sh --test

# Full reproduction (10 seeds, 4M steps, 100 episodes — hours/days on CPU)
./reproduce.sh
```

## The Environment

A parameterized variant of Gymnasium's `LunarLander-v3` with 7 continuous physics knobs that vary per episode:

| Parameter | Range | Effect |
|---|---|---|
| `gravity` | -12.0 to -2.0 | Gravitational acceleration |
| `main_engine_power` | 5.0 to 25.0 | Upward thrust scaling |
| `side_engine_power` | 0.2 to 1.5 | Lateral thrust scaling |
| `lander_density` | 2.5 to 10.0 | Lander mass (via density) |
| `angular_damping` | 0.0 to 5.0 | Rotational stability |
| `wind_power` | 0.0 to 30.0 | Horizontal wind disturbance |
| `turbulence_power` | 0.0 to 5.0 | Angular torque disturbance |

**Observation variants:**

- **Blind (8D):** Kinematic state only (position, velocity, angle, leg contact). Agent must infer physics from dynamics.
- **Labeled (15D):** Kinematic state + 7 explicit physics parameters appended to the observation vector.

**Sampling profiles** control parameter distributions: `easy` (favorable physics, no wind), `medium` (wider ranges), `hard` (extreme conditions).

## Reproducing the Claims

### Claim 1: Blind agents match or outperform labeled agents

Train 10 seeds of each variant on full-variation easy:

```bash
python lunar_lander/scripts/train_rl.py --variant blind --profile easy --seed 42 \
    --run-dir ./data/networks/full-variation/blind-ppo-easy-128-lowent/s42

python lunar_lander/scripts/train_rl.py --variant labeled --profile easy --seed 42 \
    --run-dir ./data/networks/full-variation/labeled-ppo-easy-128-lowent/s42
```

Evaluate each agent:

```bash
python lunar_lander/scripts/run_eval_pipeline.py \
    --checkpoint-dir ./data/networks/full-variation/blind-ppo-easy-128-lowent/s42 \
    --episodes 100 --trajectory-subdir trajectories
```

Compare across seeds:

```bash
python lunar_lander/scripts/compare_configs.py \
    --manifest lunar_lander/analysis-manifests/comparison/parametric-vs-behavioral.yaml
```

Look in `./data/results/parametric-vs-behavioral/stat_tests.json` for `landed_pct` p-value and Cohen's d.

### Claim 2: Blind and labeled agents develop different control strategies

The behavioral analysis produced by `run_eval_pipeline.py` reports thrust autocorrelation. Check:

```
./data/networks/full-variation/{variant}-ppo-easy-128-lowent/s{seed}/trajectories/behavioral_analysis/behavioral_summary.json
```

Compare `mean_thrust_autocorr_lag1` between blind and labeled agents. Blind agents show higher autocorrelation (smoother thrust), while labeled agents use more reactive control.

### Claim 3: Labeled agents depend on their physics labels (parametric trap)

Run corruption tests on labeled agents:

```bash
# Zero out physics labels
python lunar_lander/scripts/run_eval_pipeline.py \
    --checkpoint-dir ./data/networks/full-variation/labeled-ppo-easy-128-lowent/s42 \
    --episodes 100 --trajectory-subdir trajectories-zero \
    --corruption-mode zero

# Shuffle physics labels
python lunar_lander/scripts/run_eval_pipeline.py \
    --checkpoint-dir ./data/networks/full-variation/labeled-ppo-easy-128-lowent/s42 \
    --episodes 100 --trajectory-subdir trajectories-shuffle \
    --corruption-mode shuffle

# Replace with training-set mean
python lunar_lander/scripts/run_eval_pipeline.py \
    --checkpoint-dir ./data/networks/full-variation/labeled-ppo-easy-128-lowent/s42 \
    --episodes 100 --trajectory-subdir trajectories-mean \
    --corruption-mode mean

# Add Gaussian noise (sigma=0.1)
python lunar_lander/scripts/run_eval_pipeline.py \
    --checkpoint-dir ./data/networks/full-variation/labeled-ppo-easy-128-lowent/s42 \
    --episodes 100 --trajectory-subdir trajectories-noise \
    --corruption-mode noise --corruption-sigma 0.1
```

Compare:

```bash
python lunar_lander/scripts/compare_configs.py \
    --manifest lunar_lander/analysis-manifests/comparison/label-corruption.yaml
```

Results in `./data/results/label-corruption/stat_tests.json`.

### Claim 4: OOD gap widens under harder physics

Evaluate easy-trained agents on medium and hard profiles:

```bash
python lunar_lander/scripts/run_eval_pipeline.py \
    --checkpoint-dir ./data/networks/full-variation/blind-ppo-easy-128-lowent/s42 \
    --episodes 100 --profile medium --trajectory-subdir trajectories-ood-medium

python lunar_lander/scripts/run_eval_pipeline.py \
    --checkpoint-dir ./data/networks/full-variation/blind-ppo-easy-128-lowent/s42 \
    --episodes 100 --profile hard --trajectory-subdir trajectories-ood-hard
```

Compare:

```bash
python lunar_lander/scripts/compare_configs.py \
    --manifest lunar_lander/analysis-manifests/comparison/ood-generalization.yaml
```

Results in `./data/results/ood-generalization/stat_tests.json`.

## One-Command Reproduction

```bash
./reproduce.sh
```

Runs 6 stages via numbered bash scripts:

```
01-train.sh           Train all agent configurations (10 seeds each)
02-eval-indist.sh     In-distribution evaluation
03-eval-corruption.sh Label corruption tests (labeled agents only)
04-eval-ood.sh        Out-of-distribution evaluation (medium, hard)
05-analyze.sh         Cross-config statistical analysis
06-summary.sh         Print output file locations
```

**Quick test mode** (2 seeds, 32K steps, ~5 min):

```bash
./reproduce.sh --test
```

**Skip training** (use existing checkpoints):

```bash
./reproduce.sh --skip-training
```

**Custom output directory:**

```bash
./reproduce.sh --output-dir /path/to/output
```

**Use existing virtualenv:**

```bash
./reproduce.sh --venv /path/to/venv
```

## Running Tests

```bash
python -m pytest tests/ -v
```

Integration tests verify the full pipeline end-to-end:

- **test_single_agent_training** — trains blind + labeled agents (20K steps), verifies model output files
- **test_eval_pipeline** — collects trajectories, computes metrics, runs behavioral analysis
- **test_multi_seed_comparison** — trains 4 agents (2 variants x 2 seeds), runs `compare_configs.py`, verifies `stat_tests.json` structure

## Project Structure

```
lunar-lander-physics-representations/
├── reproduce.sh                          # One-command reproduction
├── requirements.txt
├── LICENSE
│
├── scripts/                              # Pipeline stages
│   ├── 01-train.sh
│   ├── 02-eval-indist.sh
│   ├── 03-eval-corruption.sh
│   ├── 04-eval-ood.sh
│   ├── 05-analyze.sh
│   └── 06-summary.sh
│
├── lunar_lander/
│   ├── src/                              # Core library
│   │   ├── env.py                        # Parameterized LunarLander environment
│   │   ├── physics_config.py             # 7-parameter physics configuration
│   │   ├── wrappers.py                   # PhysicsBlind, Raycast, HistoryStack
│   │   ├── label_corruption.py           # Corruption modes for eval
│   │   ├── sampling_profiles.py          # Easy/medium/hard parameter distributions
│   │   ├── training_config.py            # YAML config loader
│   │   ├── eval_utils.py                 # Evaluation utilities
│   │   ├── episode_io.py                 # Trajectory I/O (.npz)
│   │   ├── calibration.py               # Physics config calibration
│   │   ├── raycasting.py                # Terrain ray distances
│   │   ├── heuristic.py                 # Heuristic baseline agent
│   │   ├── clip_recording.py            # Video recording utilities
│   │   └── analysis/                    # Statistical analysis
│   │       ├── trajectory_metrics.py    # Per-episode metrics from .npz
│   │       ├── behavioral_metrics.py    # Behavioral characterization
│   │       ├── behavioral_comparison.py # Cross-agent behavioral analysis
│   │       ├── cross_config_comparison.py # Mann-Whitney U, Cohen's d
│   │       ├── seed_aggregation.py      # Multi-seed aggregation
│   │       ├── manifest.py              # YAML manifest loader
│   │       └── tb_parser.py             # TensorBoard log parser
│   │
│   ├── scripts/                          # Entry points
│   │   ├── train_rl.py                   # Train single agent
│   │   ├── train_all_agents.py           # Batch training from YAML configs
│   │   ├── collect_trajectories.py       # Run agent, save .npz trajectories
│   │   ├── compute_metrics.py            # .npz → metrics.csv
│   │   ├── run_eval_pipeline.py          # Collect + metrics + behavioral analysis
│   │   ├── compare_configs.py            # Cross-config statistical comparison
│   │   ├── aggregate_seeds.py            # Aggregate metrics across seeds
│   │   ├── analyze_behavior.py           # Behavioral characterization
│   │   ├── select_prototypical_episodes.py
│   │   ├── record_clips.py              # Record evaluation videos
│   │   ├── render_clips.py              # Render recorded clips
│   │   └── eval_agent.py                # Simple eval (reward only)
│   │
│   ├── configs/                          # Training configurations
│   │   ├── full-variation/               # All 7 params vary
│   │   ├── vehicle-only/                 # Fixed physics, variable vehicle
│   │   ├── physics-only/                 # Fixed vehicle, variable physics
│   │   ├── gravity-only/                 # Single-axis isolation
│   │   ├── wind-only/
│   │   ├── turbulence-only/
│   │   └── baselines/                    # Gym default (no domain randomization)
│   │
│   ├── profiles/                         # Sampling profile definitions
│   ├── batches/                          # Batch training manifests
│   └── analysis-manifests/               # Statistical comparison manifests
│       └── comparison/
│
├── rl_common/                            # Shared RL infrastructure
│   ├── training.py                       # SB3 training loop
│   ├── inference.py                      # Model loading and rollout
│   └── wrappers.py                       # Generic gym wrappers
│
├── tests/                                # Integration tests
│   ├── test_single_agent_training.py
│   ├── test_eval_pipeline.py
│   └── test_multi_seed_comparison.py
│
└── data/                                 # Generated outputs (gitignored)
    ├── networks/                         # Trained models + trajectories
    └── results/                          # Analysis outputs (stat_tests.json)
```

## License

MIT
