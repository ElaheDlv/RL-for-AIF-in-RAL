#!/usr/bin/env python3
"""
Benchmark inference latency for the available driving policies.

The script can load:
  * Active Inference (Keras) model
  * Behavioural cloning / imitation (Keras) model
  * PPO (Stable-Baselines3) policy
  * DQN (Stable-Baselines3) policy

For each supplied model it times the action-selection (forward) pass over a
single observation and reports mean / std milliseconds per call together with
steps-per-second.  Inputs default to synthetic data with the right shape but a
`.npy` file can be provided to benchmark on a recorded observation instead.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class BenchmarkResult:
    label: str
    mean_ms: float
    std_ms: float
    steps: int
    device: str
    details: str = ""


class BenchmarkError(RuntimeError):
    """Raised when a benchmark cannot be executed."""


def _load_array(path: Optional[str | Path]) -> Optional[np.ndarray]:
    if not path:
        return None
    array_path = Path(path).expanduser().resolve()
    if not array_path.exists():
        raise BenchmarkError(f"Sample array not found: {array_path}")
    if array_path.suffix != ".npy":
        raise BenchmarkError(f"Only .npy files are supported for sample inputs (got {array_path})")
    return np.load(str(array_path))


def _as_batched_array(
    array: Optional[np.ndarray],
    shape_without_batch: Tuple[int, ...],
    rng: np.random.Generator,
    *,
    low: float = 0.0,
    high: float = 1.0,
) -> np.ndarray:
    target_shape = (1,) + shape_without_batch
    if array is None:
        sample = rng.uniform(low=low, high=high, size=target_shape).astype(np.float32)
    else:
        sample = np.asarray(array, dtype=np.float32)
        if sample.shape == shape_without_batch:
            sample = sample.reshape(target_shape)
        elif sample.shape != target_shape:
            raise BenchmarkError(
                f"Sample shape {sample.shape} does not match expected {shape_without_batch} "
                "(with or without leading batch dimension)."
            )
    return sample


def _run_benchmark(
    label: str,
    call_fn: Callable[..., None],
    inputs: Sequence[np.ndarray],
    *,
    steps: int,
    warmup: int,
    sync_fn: Optional[Callable[[], None]] = None,
    device_label: str = "cpu",
    details: str = "",
) -> BenchmarkResult:
    inputs = tuple(inputs)
    for _ in range(max(warmup, 0)):
        call_fn(*inputs)
        if sync_fn:
            sync_fn()

    timings = np.empty(steps, dtype=np.float64)
    for idx in range(steps):
        start = time.perf_counter()
        call_fn(*inputs)
        if sync_fn:
            sync_fn()
        timings[idx] = time.perf_counter() - start

    mean = float(timings.mean())
    std = float(timings.std(ddof=1) if steps > 1 else 0.0)
    return BenchmarkResult(
        label=label,
        mean_ms=mean * 1000.0,
        std_ms=std * 1000.0,
        steps=steps,
        device=device_label,
        details=details,
    )


def benchmark_sb3_agent(
    *,
    label: str,
    model_path: Path,
    algo: str,
    steps: int,
    warmup: int,
    seed: int,
    sample_path: Optional[str],
) -> BenchmarkResult:
    try:
        from stable_baselines3 import DQN, PPO
        import torch as th
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise BenchmarkError(
            f"stable_baselines3 (and torch) is required to benchmark RL agents: {exc}"
        ) from exc

    algo = algo.lower()
    model_cls = PPO if algo == "ppo" else DQN if algo == "dqn" else None
    if model_cls is None:
        raise BenchmarkError(f"Unsupported SB3 algorithm '{algo}' (expected 'ppo' or 'dqn').")

    model = model_cls.load(str(model_path), device="auto")
    model.policy.set_training_mode(False)

    obs_space = model.observation_space
    if obs_space is None or obs_space.shape is None:
        raise BenchmarkError("Model does not expose an observation_space with a defined shape.")

    rng = np.random.default_rng(seed)
    sample_obs = _as_batched_array(_load_array(sample_path), obs_space.shape, rng)

    def _call(obs_batch: np.ndarray) -> None:
        obs_tensor, _ = model.policy.obs_to_tensor(obs_batch)
        with th.no_grad():
            _ = model.policy._predict(obs_tensor, deterministic=True)

    sync_fn = None
    device = model.policy.device
    if device.type == "cuda" and th.cuda.is_available():  # pragma: no branch - device check
        sync_fn = lambda: th.cuda.synchronize(device)

    details = f"obs_shape={obs_space.shape}, action_space={model.action_space}"
    return _run_benchmark(
        label=label,
        call_fn=_call,
        inputs=(sample_obs,),
        steps=steps,
        warmup=warmup,
        sync_fn=sync_fn,
        device_label=str(device),
        details=details,
    )


def benchmark_keras_model(
    *,
    label: str,
    model_path: Path,
    steps: int,
    warmup: int,
    seed: int,
    sample_paths: Optional[Sequence[Optional[str]]] = None,
) -> BenchmarkResult:
    try:
        import tensorflow as tf
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise BenchmarkError(f"TensorFlow is required to benchmark {label}: {exc}") from exc

    model = tf.keras.models.load_model(str(model_path))
    rng = np.random.default_rng(seed)

    input_tensors = model.inputs if isinstance(model.inputs, (list, tuple)) else [model.inputs]
    if not input_tensors:
        raise BenchmarkError(f"Model at {model_path} does not expose Keras inputs.")

    prepared_inputs: List[np.ndarray] = []
    shape_strings: List[str] = []
    sample_paths = sample_paths or [None] * len(input_tensors)
    if len(sample_paths) != len(input_tensors):
        raise BenchmarkError(
            f"Expected {len(input_tensors)} sample inputs, but received {len(sample_paths)}."
        )

    for idx, (tensor, path) in enumerate(zip(input_tensors, sample_paths)):
        shape = []
        for dim in tensor.shape[1:]:
            if dim is None:
                shape.append(1)
            else:
                shape.append(int(dim))
        shape_tuple = tuple(shape)

        sample = _as_batched_array(_load_array(path), shape_tuple, rng)
        prepared_inputs.append(sample)
        shape_strings.append(f"input{idx}:{shape_tuple}")

    call_fn = tf.function(lambda *args: model(*args, training=False), autograph=False)

    def _call(*arrays: np.ndarray) -> None:
        outputs = call_fn(*arrays)
        tf.nest.map_structure(
            lambda tensor: tensor.numpy() if isinstance(tensor, tf.Tensor) else tensor,
            outputs,
        )

    logical_gpus = getattr(tf.config, "list_logical_devices", lambda *_: [])("GPU")
    device_label = ",".join(dev.name for dev in logical_gpus) if logical_gpus else "cpu"

    return _run_benchmark(
        label=label,
        call_fn=_call,
        inputs=tuple(prepared_inputs),
        steps=steps,
        warmup=warmup,
        device_label=device_label,
        details=", ".join(shape_strings),
    )


def _format_results(results: List[BenchmarkResult]) -> str:
    if not results:
        return "No benchmarks were executed."

    name_width = max(len(r.label) for r in results) + 2
    header = (
        f"{'Model':<{name_width}}"
        f"{'Mean [ms]':>12}"
        f"{'Std [ms]':>12}"
        f"{'Steps/s':>12}"
        f"{'Device':>14}  Details"
    )
    lines = [header]
    for res in results:
        steps_per_s = float("inf") if res.mean_ms == 0.0 else 1000.0 / res.mean_ms
        lines.append(
            f"{res.label:<{name_width}}"
            f"{res.mean_ms:>12.3f}"
            f"{res.std_ms:>12.3f}"
            f"{steps_per_s:>12.1f}"
            f"{res.device:>14}  {res.details}"
        )
    return "\n".join(lines)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark inference latency for Active Inference, imitation, and RL agents."
    )
    parser.add_argument("--steps", type=int, default=2000, help="Timed iterations per model (default: 2000).")
    parser.add_argument("--warmup", type=int, default=50, help="Warm-up runs before timing (default: 50).")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for synthetic inputs (default: 7).")
    parser.add_argument("--export-json", type=Path, help="Optional path to write raw benchmark data as JSON.")

    parser.add_argument("--active-model", type=Path, help="Path to the Active Inference Keras model (.h5).")
    parser.add_argument("--active-road-input", help="Optional .npy file with preprocessed road mask input.")
    parser.add_argument("--active-steer-input", help="Optional .npy file with steering scalar input.")

    parser.add_argument("--imitation-model", type=Path, help="Path to the imitation learning Keras model (.h5).")
    parser.add_argument("--imitation-input", help="Optional .npy file with RGB observation input.")

    parser.add_argument("--ppo-model", type=Path, help="Path to a PPO checkpoint (.zip).")
    parser.add_argument("--ppo-input", help="Optional .npy file with PPO observation.")

    parser.add_argument("--dqn-model", type=Path, help="Path to a DQN checkpoint (.zip).")
    parser.add_argument("--dqn-input", help="Optional .npy file with DQN observation.")

    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    results: List[BenchmarkResult] = []

    if args.active_model:
        try:
            results.append(
                benchmark_keras_model(
                    label="ActiveInference",
                    model_path=args.active_model.resolve(),
                    steps=args.steps,
                    warmup=args.warmup,
                    seed=args.seed,
                    sample_paths=[args.active_road_input, args.active_steer_input],
                )
            )
        except BenchmarkError as exc:
            print(f"[ActiveInference] {exc}", file=sys.stderr)

    if args.imitation_model:
        try:
            results.append(
                benchmark_keras_model(
                    label="Imitation",
                    model_path=args.imitation_model.resolve(),
                    steps=args.steps,
                    warmup=args.warmup,
                    seed=args.seed,
                    sample_paths=[args.imitation_input],
                )
            )
        except BenchmarkError as exc:
            print(f"[Imitation] {exc}", file=sys.stderr)

    if args.ppo_model:
        try:
            results.append(
                benchmark_sb3_agent(
                    label="PPO",
                    model_path=args.ppo_model.resolve(),
                    algo="ppo",
                    steps=args.steps,
                    warmup=args.warmup,
                    seed=args.seed,
                    sample_path=args.ppo_input,
                )
            )
        except BenchmarkError as exc:
            print(f"[PPO] {exc}", file=sys.stderr)

    if args.dqn_model:
        try:
            results.append(
                benchmark_sb3_agent(
                    label="DQN",
                    model_path=args.dqn_model.resolve(),
                    algo="dqn",
                    steps=args.steps,
                    warmup=args.warmup,
                    seed=args.seed,
                    sample_path=args.dqn_input,
                )
            )
        except BenchmarkError as exc:
            print(f"[DQN] {exc}", file=sys.stderr)

    print(_format_results(results))

    if args.export_json and results:
        payload = [
            {
                "label": res.label,
                "mean_ms": res.mean_ms,
                "std_ms": res.std_ms,
                "steps": res.steps,
                "device": res.device,
                "details": res.details,
            }
            for res in results
        ]
        args.export_json.parent.mkdir(parents=True, exist_ok=True)
        args.export_json.write_text(json.dumps(payload, indent=2))
        print(f"[INFO] Wrote raw measurements to {args.export_json}")

    if not results:
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())

