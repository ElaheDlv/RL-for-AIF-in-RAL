
#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import re
from typing import Dict, List, Optional

import numpy as np
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from callbacks import DrivingMetricsCallback
from cnn_extractors import ResNetFeatureExtractor, GrayroadSmallCNN
from cnn_extractors import BCNetExtractor, GrayroadExtractor



from common_utils import (
    STEER_BINS,
    choose_policy_for_obs_space,
    read_metrics_from_info,
    StepTimer,
    LiveRenderCallback,
)
from make_env import make_env


# --- Schedules ---
def linear_schedule(start: float, end: float):
    def _fn(progress_remaining: float):  # 1.0 → 0.0
        return end + (start - end) * progress_remaining
    return _fn


def train_and_eval(env_kind: str, obs_mode: str, action_space: str,
                   timesteps: int, eval_episodes: int, seed: int,
                   town: str = None, route: str = None, out_dir: str = ".",
                   render: bool = False, render_freq: int = 1,
                   checkpoint_freq: int = 0, checkpoint_dir: Optional[str] = None,
                   ppo_kwargs: Optional[Dict] = None,
                   run_name: Optional[str] = None,
                   carla_host: str = "localhost",
                   carla_port: int = 2000):
    def _make():
        env = make_env(
            env_kind,
            obs_mode,
            action_space,
            STEER_BINS,
            seed=seed,
            show_cam=render,
            carla_host=carla_host,
            carla_port=carla_port,
        )
        return Monitor(env)
    vec = DummyVecEnv([_make])

    policy = choose_policy_for_obs_space(vec.observation_space)
    print(f"[INFO] Policy: {policy} | Obs: {obs_mode} | Action: {action_space}")

    policy_kwargs = {}
    
    # if policy == "CnnPolicy":
    #     policy_kwargs = dict(
    #         features_extractor_class=SmallCNNSB3,
    #         features_extractor_kwargs=dict(out_dim=512),
    #         net_arch=dict(pi=[256, 128], vf=[256, 128]),
    #         activation_fn=th.nn.ReLU,
    #         normalize_images=False,
    #     )
        
    
    if policy == "CnnPolicy":
        
        if obs_mode == "rgb":
        # Example for PPO with RGB input
            policy_kwargs = dict(
            features_extractor_class=BCNetExtractor,
            net_arch=dict(pi=[256,128], vf=[256,128]),
            activation_fn=th.nn.ReLU,
            )
            
        elif obs_mode == "grayroad":
        # Example for PPO with GrayRoad input
            policy_kwargs = dict(
            features_extractor_class=GrayroadExtractor,
            net_arch=dict(pi=[128,64], vf=[128,64]),
            activation_fn=th.nn.ReLU,
            )
            
        # if obs_mode == "rgb":
        #     policy_kwargs = dict(
        #         features_extractor_class=ResNetFeatureExtractor,
        #         features_extractor_kwargs=dict(out_dim=512, freeze=True),  # set freeze=False to fine-tune
        #         net_arch=dict(pi=[256, 128], vf=[256, 128]),
        #         activation_fn=th.nn.ReLU,
        #         normalize_images=False,
        #         )
        # elif obs_mode == "grayroad":
        #     policy_kwargs = dict(
        #         features_extractor_class=GrayroadSmallCNN,
        #         features_extractor_kwargs=dict(out_dim=256),
        #         net_arch=dict(pi=[128, 64], vf=[128, 64]),
        #         activation_fn=th.nn.ReLU,
        #         normalize_images=False,
        #     )   


    lr_schedule   = linear_schedule(1e-4, 5e-5)
    clip_schedule = linear_schedule(0.20, 0.10)

    def _sanitize_tag(text: str) -> str:
        text = text.strip()
        text = re.sub(r"[^0-9a-zA-Z._-]+", "-", text)
        return text.strip("-_") or "run"

    def _config_suffix(config: Optional[Dict]) -> str:
        if not config:
            return "cfg-default"
        try:
            payload = json.dumps(config, sort_keys=True, default=str)
        except TypeError:
            payload = repr(config)
        digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:8]
        return f"cfg-{digest}"

    def _merge_policy_kwargs(base: Optional[Dict], override: Optional[Dict]) -> Optional[Dict]:
        if base is None:
            base = {}
        if not override:
            return base if base else None
        merged = dict(base)
        for key, value in override.items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                inner = dict(merged[key])
                inner.update(value)
                merged[key] = inner
            else:
                merged[key] = value
        return merged

    config_suffix = _config_suffix(ppo_kwargs)
    if run_name:
        log_tag = _sanitize_tag(run_name)
    else:
        log_tag = _sanitize_tag(
            f"{env_kind}-{obs_mode}-{action_space}-steps{timesteps}-seed{seed}-{config_suffix}"
        )

    tensorboard_dir = os.path.join(out_dir, "tb_logs", log_tag)
    os.makedirs(os.path.dirname(tensorboard_dir), exist_ok=True)

    algo_kwargs = dict(
        policy=policy,
        env=vec,
        verbose=1,
        seed=seed,
        n_steps=1024,
        batch_size=64,
        gae_lambda=0.95,
        gamma=0.99,
        n_epochs=10,
        learning_rate=lr_schedule,
        clip_range=clip_schedule,
        target_kl=0.02,          # guardrail on policy updates
        ent_coef=0.01,
        tensorboard_log=tensorboard_dir,
    )
    if policy_kwargs:
        algo_kwargs["policy_kwargs"] = policy_kwargs

    if ppo_kwargs:
        ppo_kwargs = dict(ppo_kwargs)  # shallow copy so callers can reuse dicts
        policy_kw_override = ppo_kwargs.pop("policy_kwargs", None)
        if policy_kw_override:
            existing = algo_kwargs.get("policy_kwargs")
            algo_kwargs["policy_kwargs"] = _merge_policy_kwargs(existing, policy_kw_override)
        algo_kwargs.update(ppo_kwargs)

    def _normalize_policy_kwargs(kwargs: Optional[Dict]) -> Optional[Dict]:
        if not kwargs:
            return kwargs
        extractor = kwargs.get("features_extractor_class")
        if isinstance(extractor, str):
            alias_map = {
                "GrayroadSmallCNN": GrayroadSmallCNN,
                "GrayroadExtractor": GrayroadSmallCNN,
                "ResNetFeatureExtractor": ResNetFeatureExtractor,
            }
            resolved = alias_map.get(extractor)
            if resolved is None:
                resolved = globals().get(extractor)
            if resolved is None:
                raise ValueError(f"Unknown features_extractor_class '{extractor}'")
            kwargs["features_extractor_class"] = resolved
        return kwargs

    normalized_policy_kwargs = _normalize_policy_kwargs(algo_kwargs.get("policy_kwargs"))
    if normalized_policy_kwargs:
        algo_kwargs["policy_kwargs"] = normalized_policy_kwargs
    else:
        algo_kwargs.pop("policy_kwargs", None)

    model = PPO(**algo_kwargs)

    callbacks = []
    if render:
        callbacks.append(LiveRenderCallback(vec, freq=render_freq))
    if checkpoint_freq > 0:
        save_dir = checkpoint_dir or os.path.join(out_dir, "checkpoints")
        os.makedirs(save_dir, exist_ok=True)
        callbacks.append(
            CheckpointCallback(
                save_freq=int(checkpoint_freq),
                save_path=save_dir,
                name_prefix=f"ppo_{env_kind}_{obs_mode}_{action_space}"
            )
        )
    callbacks.append(DrivingMetricsCallback())
    callback_list = CallbackList(callbacks) if callbacks else None

    if timesteps % model.n_steps != 0:
        rounded = (timesteps // model.n_steps + 1) * model.n_steps
        print(
            f"[WARN] total_timesteps={timesteps} is not a multiple of rollout n_steps={model.n_steps}. "
            f"SB3 will collect {rounded} steps."
        )

    model.learn(total_timesteps=timesteps, progress_bar=True, callback=callback_list)

    if hasattr(vec, "close"):
        vec.close()


    save_tag = f"ppo_{log_tag}.zip"
    save_path = os.path.join(out_dir, save_tag)
    model.save(save_path)
    print(f"[INFO] Saved model to {save_path}")

    # Evaluation
    single_env = make_env(
        env_kind,
        obs_mode,
        action_space,
        STEER_BINS,
        seed=seed + 123,
        show_cam=False,
        carla_host=carla_host,
        carla_port=carla_port,
    )
    successes, deviations = [], []
    timer = StepTimer()

    for ep in range(eval_episodes):
        obs, info = single_env.reset(seed=seed+1000+ep)
        terminated = truncated = False
        ep_devs: List[float] = []
        while not (terminated or truncated):
            timer.start()
            action, _ = model.predict(obs, deterministic=True)
            timer.stop()
            obs, reward, terminated, truncated, info = single_env.step(action if action_space=="disc" else np.array(action))
            if "lane_deviation" in info:
                ep_devs.append(float(info["lane_deviation"]))
        succ, dev = read_metrics_from_info(info)
        if succ is None:
            succ = not terminated
        if dev is None:
            dev = float(np.mean(ep_devs)) if ep_devs else np.nan
        successes.append(bool(succ))
        deviations.append(float(dev))

    if hasattr(single_env, "close"):
        single_env.close()

    success_rate = 100.0 * (np.sum(successes) / len(successes))
    avg_dev = float(np.nanmean(deviations)) if len(deviations) else float("nan")
    print("======== Evaluation ========")
    print(f"Success Rate: {success_rate:.2f} %")
    print(f"Avg. Deviation: {avg_dev:.4f} m")
    print(f"Inference FPS (mean): {timer.mean_fps:.2f}")
    print(f"Latency per step (ms): {timer.mean_latency_ms:.3f}")
    print("============================")

    # CSV log
    import csv
    csv_path = os.path.join(out_dir, "results_ppo.csv")
    header = ["method","obs_mode","action_space","success_rate","avg_deviation_m","fps","latency_ms","timesteps"]
    row = ["PPO", obs_mode, action_space, f"{success_rate:.2f}", f"{avg_dev:.4f}", f"{timer.mean_fps:.2f}", f"{timer.mean_latency_ms:.3f}", timesteps]
    exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(header)
        w.writerow(row)
    print(f"[INFO] Appended results to {csv_path}")

    return {
        "success_rate": success_rate,
        "avg_deviation_m": avg_dev,
        "fps": timer.mean_fps,
        "latency_ms": timer.mean_latency_ms,
        "timesteps": timesteps,
        "log_tag": log_tag,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="carla", choices=["carla","toy"])
    ap.add_argument("--obs", default="rgb", choices=["rgb","grayroad"])
    ap.add_argument("--action", default="disc", choices=["disc","cont"])
    ap.add_argument("--timesteps", type=int, default=2_000_000)
    ap.add_argument("--eval-episodes", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=".")
    ap.add_argument("--render", action="store_true", help="Show live camera feed during training")
    ap.add_argument("--render-freq", type=int, default=1, help="Render every N environment steps (>=1)")
    ap.add_argument("--checkpoint-freq", type=int, default=100_000, help="Save model every N steps (0 disables)")
    ap.add_argument("--checkpoint-dir", default=None, help="Directory for checkpoints (defaults to <out>/checkpoints)")
    ap.add_argument("--carla-host", default="localhost", help="CARLA server host")
    ap.add_argument("--carla-port", type=int, default=2000, help="CARLA server port")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    checkpoint_dir = args.checkpoint_dir or os.path.join(args.out, "checkpoints")
    train_and_eval(
        args.env,
        args.obs,
        args.action,
        args.timesteps,
        args.eval_episodes,
        args.seed,
        out_dir=args.out,
        render=args.render,
        render_freq=args.render_freq,
        checkpoint_freq=args.checkpoint_freq,
        checkpoint_dir=checkpoint_dir,
        carla_host=args.carla_host,
        carla_port=args.carla_port,
    )

if __name__ == "__main__":
    main()
