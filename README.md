
# RL Baselines for Lane-Keeping Comparisons

This package gives you *four PPO variants* and an optional *DQN-Disc* to compare against your AIF and BC:
- PPO-Disc + **RGB** (ResNet18 pretrained on ImageNet as encoder)
- PPO-Disc + **GrayRoad** (lightweight CNN trained from scratch)
- PPO-Cont + **RGB** (ResNet18 pretrained encoder)
- PPO-Cont + **GrayRoad** (lightweight CNN)
- (Optional) DQN-Disc + RGB/GrayRoad (uses the same extractors as PPO)


It prefers your CARLA `CarEnv`; if you want a lightweight smoke test, launch with `--env toy` explicitly. The CARLA option now fails fast when the simulator is unavailable or incompatible so you don't silently train on the toy env.

## Install
```bash
conda create -n rl-baselines python=3.10 -y
conda activate rl-baselines
pip install -r requirements.txt
```

## Usage (PPO)
```bash
# Discrete 20-bin steering, RGB
python train_ppo.py --env carla --obs rgb --action disc --timesteps 2000000 --eval-episodes 12 --out runs/ppo_disc_rgb

# Discrete 20-bin steering, GrayRoad
python train_ppo.py --env carla --obs grayroad --action disc --timesteps 2000000 --eval-episodes 12 --out runs/ppo_disc_gray

# Continuous steering in [-1,1], RGB
python train_ppo.py --env carla --obs rgb --action cont --timesteps 2000000 --eval-episodes 12 --out runs/ppo_cont_rgb

# Continuous steering in [-1,1], GrayRoad
python train_ppo.py --env carla --obs grayroad --action cont --timesteps 2000000 --eval-episodes 12 --out runs/ppo_cont_gray

python run_rl_sweeps_with_config.py --config sweep_rgb.json --out runs/sweeps_rgb
python run_rl_sweeps_with_config.py --config sweep_gray.json --out runs/sweeps_gray

# Add `--render --render-freq 1` to any command above to watch the live CARLA camera during training.
# Use `--carla-host` / `--carla-port` if your simulator runs on a non-default address.
# Sweeps default to checkpointing every 100k steps; override with `--checkpoint-freq`.
# Resume a stopped run with `--resume-from path/to/model.zip`.

python train_ppo.py --env carla --obs rgb --action disc --timesteps 8000000 --eval-episodes 12 --out runs/ppo_disc_rgb --render --render-freq 1

python run_rl_sweeps.py --checkpoint-freq 100000 --checkpoint-dir checkpoints

```
Note: For `--obs rgb`, PPO now uses a ResNet18 backbone pretrained on ImageNet (with frozen early layers).
For `--obs grayroad`, PPO uses a smaller custom CNN trained from scratch, optimized for binary road masks.
Also `--render --render-freq 1` to any command above to watch the live CARLA camera during training.


## Usage (DQN-Disc)
```bash
python train_dqn_disc.py --env carla --obs rgb --timesteps 300000 --eval-episodes 12 --out runs/dqn_disc_rgb
python train_dqn_disc.py --env carla --obs grayroad --timesteps 300000 --eval-episodes 12 --out runs/dqn_disc_gray
# Pass `--render --render-freq 1` here as well for a real-time view, and `--carla-host` / `--carla-port`
# to point at a remote CARLA server when needed. Sweeps place intermediate checkpoints in `checkpoints/`
# under each run directory by default.
# Use `python run_rl_sweeps_with_config.py --resume-existing ...` to continue unfinished sweeps (or rerun individual training scripts with `--resume-from`).
```

## Route Testing

After training, evaluate each method on specific CARLA spawn pairs. Provide routes inline as `start_idx:goal_idx` or via a JSON file containing a `routes` list (e.g. `{"routes": [{"start": 416, "goal": 252}]}`).

### RL policies (PPO / DQN)
```bash
# PPO discrete steering, RGB observations
python test_rl_routes.py --algo ppo --model runs/ppo_disc_rgb/ppo_carla_rgb_disc.zip \
    --obs rgb --action disc --routes 416:252 120:45 --render --log-dir logs/ppo_disc_rgb

# PPO discrete steering, GrayRoad observations
python test_rl_routes.py --algo ppo --model runs/ppo_disc_gray/ppo_carla_grayroad_disc.zip \
    --obs grayroad --action disc --routes 416:252 120:45 --log-dir logs/ppo_disc_gray

# PPO continuous steering, RGB observations
python test_rl_routes.py --algo ppo --model runs/ppo_cont_rgb/ppo_carla_rgb_cont.zip \
    --obs rgb --action cont --routes 416:252 120:45 --log-dir logs/ppo_cont_rgb

# PPO continuous steering, GrayRoad observations
python test_rl_routes.py --algo ppo --model runs/ppo_cont_gray/ppo_carla_grayroad_cont.zip \
    --obs grayroad --action cont --routes 416:252 120:45 --log-dir logs/ppo_cont_gray

# DQN discrete steering, RGB observations (using JSON route list)
python test_rl_routes.py --algo dqn --model runs/dqn_disc_rgb/dqn_carla_rgb_disc.zip \
    --routes-json routes.json --render --log-dir logs/dqn_disc_rgb

# DQN discrete steering, GrayRoad observations
python test_rl_routes.py --algo dqn --model runs/dqn_disc_gray/dqn_carla_grayroad_disc.zip \
    --obs grayroad --action disc --routes 416:252 120:45 --log-dir logs/dqn_disc_gray
# Append `--carla-host HOST --carla-port PORT` to any command above if the CARLA simulator
# listens on a non-default endpoint (defaults remain `localhost:2000`).
```

### Active Inference baseline
```bash
python test_active_inference.py --model Active_inference_trained_model.h5 --reference ref6.png \
    --routes 416:252 120:45 --csv results_active_inference.csv --log-dir logs/aif_eval
```

### Imitation learning baseline
```bash
python test_imitation_agent.py --model Imitation_learning_trained_model.h5 --routes 416:252 120:45 \
    --csv results_imitation.csv --log-dir logs/bc_eval
```

All evaluation scripts stop runs early if the vehicle stays under 1 km/h for more than five minutes or starts moving away from the goal after getting close. The optional `--log-dir` trajectory files capture per-step pose, speed, and lane deviation so you can recompute metrics such as average deviation offline.

### Note on your `CarEnv`
We attempt `from car_env import CarEnv`. Your env should accept a config dict with keys like:
```python
{
  "image_shape": (160,160,1) or (160,160,3),
  "obs_mode": "grayroad" or "rgb",
  "discrete_actions": True/False,
  "steer_bins": [-1.0, ..., 1.0],   # when discrete
  "action_space": "continuous",     # when continuous
  "max_steps": 600,
  "seed": 0,
}
```
At episode end, please provide in `info`:
- `success` (bool) and
- `lane_deviation` (float, meters).

If you don't have these, the scripts still run but success/dev may fall back to rough proxies.

### Outputs
- SB3 model files in your `--out` folder
- A CSV summary `results_ppo.csv` / `results_dqn.csv` with:
  `method, obs_mode, action_space, success_rate, avg_deviation_m, fps, latency_ms, timesteps`

### Tips
- Match BC: use **RGB + continuous** to mirror BC.
- Match AIF: use **GrayRoad + discrete** (20 bins) to mirror AIF.
- Report runtime: FPS & latency are printed and logged.
