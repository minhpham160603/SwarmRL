import sys

sys.path.append("../")
import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3 import PPO, SAC, A2C
import gymnasium as gym
import swarm_env.single_env  # import to use with gym.make()
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import (
    VecVideoRecorder,
    SubprocVecEnv,
)
from stable_baselines3.common.callbacks import CallbackList
import numpy as np
from utils import EpisodicRewardLogger, DummyRun, AverageReturnCallback
import torch.nn as nn
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from datetime import datetime
import torch
import random


using_wandb = False

config = {
    "algo": "PPO",
    "total_timesteps": 100_000,
    "max_steps": 100,
    "num_envs": 4,
}

env_config = {
    "id": "SwarmEnv-v0",
    "max_steps": 100,
    "map_name": "Easy",
    "continuous_action": True,
    "fixed_step": 20,
}

kwargs_policy = {
    "policy": "MultiInputPolicy",
    "policy_kwargs": {
        "net_arch": {"pi": [16, 32, 64, 32, 16], "vf": [16, 32, 64, 32, 16]},
        "activation_fn": nn.ReLU,
        "ortho_init": True,
    },
    "verbose": 1,
}

wandb_config = {
    "exp_name": "test_step",
    "map": env_config["map_name"],
    "max_step": env_config["max_steps"],
    "fix_step": env_config["fixed_step"],
    "algo": config["algo"],
    "policy_kwarg": (
        kwargs_policy["policy_kwargs"] if "policy_kwargs" in kwargs_policy else None
    ),
    "map_size": (300, 300),
    "min_gen_dist": 120,
}

# Get today's date
today = datetime.now()
formatted_date = today.strftime("%d-%m")


def make_env():
    env = gym.make(**env_config)
    env = Monitor(env)
    return env


seed = 1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

env = SubprocVecEnv([make_env for i in range(config["num_envs"])], start_method="fork")


if using_wandb:
    run = wandb.init(
        project="single-env-v2",
        config=wandb_config,
        sync_tensorboard=True,
        monitor_gym=True,
        save_code=True,
        name=f"{wandb_config['exp_name']}_{config['algo']}_seed{seed}",
        # group=config["algo"],
    )
else:
    run = DummyRun()

# env = VecVideoRecorder(
#     env,
#     f"videos/{formatted_date}/{run.id}",
#     record_video_trigger=lambda x: x % 5_000 == 0,
#     video_length=config["max_steps"],
# )

average_callback = AverageReturnCallback(verbose=1, n_episodes=100)
wandbcallback = (
    WandbCallback(
        gradient_save_freq=0,
        model_save_path=f"models/{formatted_date}/{run.id}",
        verbose=2,
    )
    if using_wandb
    else None
)

# Save a checkpoint every 1000 steps
checkpoint_callback = CheckpointCallback(
    save_freq=5_000,
    save_path=f"./checkpoints/{formatted_date}/{run.id}",
    name_prefix=f"model_{run.id}",
    save_replay_buffer=True,
    save_vecnormalize=True,
)

eval_callback = EvalCallback(
    env,
    best_model_save_path=f"./logs/{formatted_date}/{run.id}",
    log_path=f"./logs/{formatted_date}/{run.id}",
    eval_freq=10_000,
    deterministic=True,
    render=False,
)


algo_map = {"PPO": PPO, "SAC": SAC, "A2C": A2C, "R_PPO": RecurrentPPO}
model = algo_map[config["algo"]](
    env=env, tensorboard_log=f"runs/{formatted_date}/{run.id}", **kwargs_policy
)

callbacklist = [average_callback, eval_callback]
if using_wandb:
    callbacklist.append(wandbcallback)
callbacklist = CallbackList(callbacklist)

print("ALGO ", config["algo"])
print(model.policy)

model.learn(
    total_timesteps=config["total_timesteps"],
    callback=callbacklist,
    progress_bar=True,
)

env.close()
run.finish()
