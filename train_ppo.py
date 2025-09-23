
#!/usr/bin/env python3
import argparse
import os
from typing import List, Optional

import numpy as np
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from callbacks import DrivingMetricsCallback
from cnn_extractors import SmallCNNSB3
from common_utils import (
    STEER_BINS,
    choose_policy_for_obs_space,
    read_metrics_from_info,
    StepTimer,
    LiveRenderCallback,
)
from make_env import make_env


def train_and_eval(env_kind: str, obs_mode: str, action_space: str,
                   timesteps: int, eval_episodes: int, seed: int,
                   town: str = None, route: str = None, out_dir: str = ".",
                   render: bool = False, render_freq: int = 1,
                   checkpoint_freq: int = 0, checkpoint_dir: Optional[str] = None):
    def _make():
        return make_env(env_kind, obs_mode, action_space, STEER_BINS, seed=seed, show_cam=render)
    vec = DummyVecEnv([_make])

    policy = choose_policy_for_obs_space(vec.observation_space)
    print(f"[INFO] Policy: {policy} | Obs: {obs_mode} | Action: {action_space}")

    policy_kwargs = {}
    if policy == "CnnPolicy":
        policy_kwargs = dict(
            features_extractor_class=SmallCNNSB3,
            features_extractor_kwargs=dict(out_dim=512),
            net_arch=dict(pi=[256, 128], vf=[256, 128]),
            activation_fn=th.nn.ReLU,
            normalize_images=False,
        )

    model = PPO(
        policy,
        vec,
        verbose=1,
        seed=seed,
        n_steps=2048,
        batch_size=32,
        gae_lambda=0.95,
        gamma=0.99,
        n_epochs=10,
        learning_rate=3e-4,
        clip_range=0.2,
        tensorboard_log=os.path.join(out_dir, "tb_logs_ppo"),
        policy_kwargs=policy_kwargs if policy_kwargs else None,
    )
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


    save_tag = f"ppo_{env_kind}_{obs_mode}_{action_space}.zip"
    save_path = os.path.join(out_dir, save_tag)
    model.save(save_path)
    print(f"[INFO] Saved model to {save_path}")

    # Evaluation
    single_env = make_env(env_kind, obs_mode, action_space, STEER_BINS, seed=seed+123, show_cam=False)
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
    )

if __name__ == "__main__":
    main()
