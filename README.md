
# RL Baselines for Lane-Keeping Comparisons

This package gives you *four PPO variants* and an optional *DQN-Disc* to compare against your AIF and BC:
- PPO-Disc + **RGB**
- PPO-Disc + **GrayRoad** (road-only grayscale like AIF)
- PPO-Cont + **RGB**
- PPO-Cont + **GrayRoad**
- (Optional) **DQN-Disc** + RGB/GrayRoad

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
python train_ppo.py --env carla --obs rgb --action disc --timesteps 300000 --eval-episodes 12 --out runs/ppo_disc_rgb

# Discrete 20-bin steering, GrayRoad
python train_ppo.py --env carla --obs grayroad --action disc --timesteps 300000 --eval-episodes 12 --out runs/ppo_disc_gray

# Continuous steering in [-1,1], RGB
python train_ppo.py --env carla --obs rgb --action cont --timesteps 300000 --eval-episodes 12 --out runs/ppo_cont_rgb

# Continuous steering in [-1,1], GrayRoad
python train_ppo.py --env carla --obs grayroad --action cont --timesteps 300000 --eval-episodes 12 --out runs/ppo_cont_gray
\n+# Add `--render --render-freq 1` to any command above to watch the live CARLA camera during training.
```

## Usage (DQN-Disc)
```bash
python train_dqn_disc.py --env carla --obs rgb --timesteps 300000 --eval-episodes 12 --out runs/dqn_disc_rgb
python train_dqn_disc.py --env carla --obs grayroad --timesteps 300000 --eval-episodes 12 --out runs/dqn_disc_gray
\n+# Pass `--render --render-freq 1` here as well for a real-time view.
```

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
