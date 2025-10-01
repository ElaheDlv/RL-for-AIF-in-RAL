#!/usr/bin/env python3
"""Utility to launch a batch of PPO/DQN training runs with preset hyperparameters.

Each experiment gets its own output directory under ``--out`` and training metrics
are appended to a top-level ``summary.csv`` for quick comparison.
"""

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import math


def _sanitize_tag(text: str) -> str:
    import re

    text = text.strip()
    text = re.sub(r"[^0-9a-zA-Z._-]+", "-", text)
    return text.strip("-_") or "run"


def _hash_config(config: Dict[str, Any]) -> str:
    try:
        payload = json.dumps(config, sort_keys=True, default=str)
    except TypeError:
        payload = repr(config)
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:8]
    return digest


def _fmt_float(value: Optional[float], fmt: str) -> str:
    if value is None:
        return ""
    if isinstance(value, (int, float)):
        if isinstance(value, float) and (not math.isfinite(value)):
            return ""
        return format(value, fmt)
    return str(value)

from train_ppo import train_and_eval as train_ppo_and_eval
from train_dqn_disc import train_and_eval as train_dqn_and_eval


@dataclass
class Experiment:
    name: str
    algo: str  # "ppo" or "dqn"
    obs_mode: str
    action: str = "disc"
    env_kind: str = "carla"
    timesteps: int = 4_000_000
    eval_episodes: int = 12
    seed: int = 0
    checkpoint_freq: int = 0
    algo_kwargs: Dict[str, Any] = field(default_factory=dict)
    notes: str = ""
    render: bool = True
    render_freq: int = 1
    carla_host: str = "localhost"
    carla_port: int = 2000


SUMMARY_FIELDS = [
    "timestamp",
    "experiment",
    "log_tag",
    "algo",
    "env",
    "obs_mode",
    "action",
    "render",
    "render_freq",
    "carla_host",
    "carla_port",
    "timesteps",
    "eval_episodes",
    "seed",
    "success_rate",
    "avg_deviation_m",
    "fps",
    "latency_ms",
    "run_dir",
    "hyperparams",
    "notes",
]


def _load_last_result(csv_path: Path) -> Dict[str, str]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing results file: {csv_path}")
    with csv_path.open("r", newline="") as f:
        reader = list(csv.reader(f))
    if len(reader) < 2:
        raise RuntimeError(f"No results rows found in {csv_path}")
    header = reader[0]
    row = reader[-1]
    return dict(zip(header, row))


def _dump_summary(summary_path: Path, row: Dict[str, Any]) -> None:
    exists = summary_path.exists()
    with summary_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, SUMMARY_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _json_dumps(data: Dict[str, Any]) -> str:
    if not data:
        return "{}"
    return json.dumps(data, sort_keys=True, default=str)


# --- Define your sweep here ---
EXPERIMENTS: List[Experiment] = [
    # PPO on RGB: longer training baseline (defaults but 4M steps)
    Experiment(
        name="ppo_rgb_disc_long_baseline",
        algo="ppo",
        obs_mode="rgb",
        action="disc",
        timesteps=4_000_000,
        notes="Default hyperparameters, extended training horizon",
    ),
    # PPO on RGB: larger batches, unfreeze ResNet
    Experiment(
        name="ppo_rgb_disc_large_batch_unfreeze",
        algo="ppo",
        obs_mode="rgb",
        action="disc",
        timesteps=4_000_000,
        algo_kwargs={
            "n_steps": 2048,
            "batch_size": 256,
            "n_epochs": 6,
            "gae_lambda": 0.97,
            "target_kl": 0.03,
            "learning_rate": 3e-4,
            "policy_kwargs": {
                "features_extractor_kwargs": {"freeze": False},
                "net_arch": dict(pi=[512, 256], vf=[512, 256]),
            },
        },
        notes="Bigger rollout batch and policy head, ResNet fine-tuning enabled",
    ),
    # PPO on grayscale: deeper CNN and more entropy regularisation
    Experiment(
        name="ppo_gray_disc_deeper_entropy",
        algo="ppo",
        obs_mode="grayroad",
        action="disc",
        timesteps=3_000_000,
        algo_kwargs={
            "n_steps": 1536,
            "batch_size": 128,
            "learning_rate": 7e-5,
            "gae_lambda": 0.97,
            "ent_coef": 0.02,
            "policy_kwargs": {
                "features_extractor_kwargs": {"out_dim": 384},
                "net_arch": dict(pi=[256, 128], vf=[256, 128]),
            },
        },
        notes="Gray-road encoder widened with higher entropy weight",
    ),
    # PPO continuous control example for RGB
    Experiment(
        name="ppo_rgb_cont_long_clip10",
        algo="ppo",
        obs_mode="rgb",
        action="cont",
        timesteps=4_000_000,
        algo_kwargs={
            "n_steps": 1024,
            "batch_size": 128,
            "clip_range": 0.1,
            "learning_rate": 1.5e-4,
            "gae_lambda": 0.9,
        },
        notes="Continuous steering with tighter clip range",
    ),
    # DQN on RGB: larger replay and slower epsilon decay
    Experiment(
        name="dqn_rgb_large_buffer_slow_eps",
        algo="dqn",
        obs_mode="rgb",
        timesteps=2_500_000,
        algo_kwargs={
            "learning_rate": 5e-4,
            "buffer_size": 200_000,
            "learning_starts": 20_000,
            "target_update_interval": 2_000,
            "exploration_fraction": 0.6,
            "policy_kwargs": {
                "net_arch": [512, 256],
            },
        },
        notes="RGB DQN with larger network and replay buffer",
    ),
    # DQN on GrayRoad: lower LR, deeper head, faster target sync
    Experiment(
        name="dqn_gray_low_lr_fast_target",
        algo="dqn",
        obs_mode="grayroad",
        timesteps=2_000_000,
        algo_kwargs={
            "learning_rate": 2e-4,
            "train_freq": 8,
            "target_update_interval": 2_500,
            "exploration_fraction": 0.35,
            "exploration_final_eps": 0.02,
            "policy_kwargs": {
                "features_extractor_kwargs": {"out_dim": 384},
                "net_arch": [256, 128],
            },
        },
        notes="Gray-road DQN with wider features and tighter epsilon",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a suite of RL training experiments")
    parser.add_argument("--out", default="runs/sweeps", help="Directory to store experiment outputs")
    parser.add_argument("--env", default=None, help="Override env kind for all experiments (default: each experiment setting)")
    parser.add_argument(
        "--only",
        nargs="*",
        help="Subset of experiment names to run",
    )
    parser.add_argument("--list", action="store_true", help="List available experiments and exit")
    parser.add_argument("--skip-existing", action="store_true", help="Skip runs whose results file already exists")
    parser.add_argument("--render", dest="render", action="store_true", help="Force rendering on for all experiments")
    parser.add_argument("--no-render", dest="render", action="store_false", help="Disable rendering for all experiments")
    parser.set_defaults(render=None)
    parser.add_argument("--render-freq", type=int, default=None, help="Override render frequency for all experiments")
    parser.add_argument("--carla-host", default=None, help="Override CARLA host for all experiments")
    parser.add_argument("--carla-port", type=int, default=None, help="Override CARLA port for all experiments")
    return parser.parse_args()


def build_run_tag(
    exp: Experiment,
    env_kind: str,
    render: bool,
    render_freq: int,
    carla_host: str,
    carla_port: int,
) -> str:
    parts = [
        _sanitize_tag(exp.name),
        exp.algo,
        f"obs-{exp.obs_mode}",
        f"act-{exp.action}",
        f"env-{env_kind}",
        f"steps-{exp.timesteps}",
        f"seed-{exp.seed}",
        f"render-{int(bool(render))}",
        f"rfreq-{render_freq}",
        f"host-{_sanitize_tag(carla_host)}",
        f"port-{carla_port}",
    ]
    if exp.algo_kwargs:
        parts.append(f"cfg-{_hash_config(exp.algo_kwargs)}")
    return _sanitize_tag("__".join(parts))


def main() -> None:
    args = parse_args()
    if args.config:
        experiments_source = _load_experiments_from_config(Path(args.config))
    else:
        experiments_source = list(EXPERIMENTS)

    if args.list:
        for exp in experiments_source:
            print(
                f"{exp.name}: {exp.algo} | obs={exp.obs_mode} | action={exp.action} | steps={exp.timesteps}"
            )
        return

    selected = {name for name in (args.only or [])}
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    summary_path = out_root / "summary.csv"

    for exp in experiments_source:
        if selected and exp.name not in selected:
            continue

        env_kind = args.env or exp.env_kind
        render_flag = exp.render if args.render is None else args.render
        render_freq = exp.render_freq if args.render_freq is None else max(1, args.render_freq)
        carla_host = args.carla_host if args.carla_host is not None else exp.carla_host
        carla_port = args.carla_port if args.carla_port is not None else exp.carla_port

        run_tag = build_run_tag(exp, env_kind, render_flag, render_freq, carla_host, carla_port)
        run_dir = out_root / run_tag
        run_dir.mkdir(parents=True, exist_ok=True)

        results_file = run_dir / ("results_ppo.csv" if exp.algo == "ppo" else "results_dqn.csv")
        if args.skip_existing and results_file.exists():
            print(f"[SKIP] {exp.name} (results already present)")
            continue

        print(f"[RUN] {exp.name} → {run_dir}")

        metrics = None
        try:
            config_manifest = {
                "experiment": exp.name,
                "run_tag": run_tag,
                "algo": exp.algo,
                "env": env_kind,
                "obs_mode": exp.obs_mode,
                "action": exp.action,
                "timesteps": exp.timesteps,
                "eval_episodes": exp.eval_episodes,
                "seed": exp.seed,
                "checkpoint_freq": exp.checkpoint_freq,
                "algo_kwargs": exp.algo_kwargs,
                "render": render_flag,
                "render_freq": render_freq,
                "carla_host": carla_host,
                "carla_port": carla_port,
                "notes": exp.notes,
            }
            with (run_dir / "config.json").open("w") as cfg_f:
                json.dump(config_manifest, cfg_f, indent=2, default=str)

            if exp.algo == "ppo":
                metrics = train_ppo_and_eval(
                    env_kind,
                    exp.obs_mode,
                    exp.action,
                    exp.timesteps,
                    exp.eval_episodes,
                    exp.seed,
                    out_dir=str(run_dir),
                    checkpoint_freq=exp.checkpoint_freq,
                    ppo_kwargs=exp.algo_kwargs,
                    render=render_flag,
                    render_freq=render_freq,
                    run_name=run_tag,
                    carla_host=carla_host,
                    carla_port=carla_port,
                )
            elif exp.algo == "dqn":
                metrics = train_dqn_and_eval(
                    env_kind,
                    exp.obs_mode,
                    exp.timesteps,
                    exp.eval_episodes,
                    exp.seed,
                    out_dir=str(run_dir),
                    checkpoint_freq=exp.checkpoint_freq,
                    dqn_kwargs=exp.algo_kwargs,
                    render=render_flag,
                    render_freq=render_freq,
                    run_name=run_tag,
                    carla_host=carla_host,
                    carla_port=carla_port,
                )
            else:
                raise ValueError(f"Unknown algo '{exp.algo}' in experiment {exp.name}")
        except Exception as exc:
            print(f"[FAIL] {exp.name}: {exc}")
            continue

        if not metrics:
            try:
                result = _load_last_result(results_file)
            except Exception as exc:  # noqa: PIE786 - keep context in message
                print(f"[WARN] Could not parse results for {exp.name}: {exc}")
                continue
            metrics = {
                "success_rate": float(result.get("success_rate", "nan")),
                "avg_deviation_m": float(result.get("avg_deviation_m", "nan")),
                "fps": float(result.get("fps", "nan")),
                "latency_ms": float(result.get("latency_ms", "nan")),
                "timesteps": int(result.get("timesteps", 0)),
                "log_tag": run_tag,
            }

        summary_row = {
            "timestamp": datetime.utcnow().isoformat(timespec="seconds"),
            "experiment": exp.name,
            "log_tag": metrics.get("log_tag", run_tag),
            "algo": exp.algo,
            "env": env_kind,
            "obs_mode": exp.obs_mode,
            "action": exp.action,
            "render": "true" if render_flag else "false",
            "render_freq": str(render_freq),
            "carla_host": carla_host,
            "carla_port": str(carla_port),
            "timesteps": str(metrics.get("timesteps", exp.timesteps)),
            "eval_episodes": str(exp.eval_episodes),
            "seed": str(exp.seed),
            "success_rate": _fmt_float(metrics.get("success_rate"), ".2f"),
            "avg_deviation_m": _fmt_float(metrics.get("avg_deviation_m"), ".4f"),
            "fps": _fmt_float(metrics.get("fps"), ".2f"),
            "latency_ms": _fmt_float(metrics.get("latency_ms"), ".3f"),
            "run_dir": str(run_dir),
            "hyperparams": _json_dumps(exp.algo_kwargs),
            "notes": exp.notes,
        }
        _dump_summary(summary_path, summary_row)
        print(f"[DONE] {exp.name} → success_rate={summary_row['success_rate']} avg_dev={summary_row['avg_deviation_m']}")


if __name__ == "__main__":
    main()
