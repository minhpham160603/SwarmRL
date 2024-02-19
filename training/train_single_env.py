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
from utils import EpisodicRewardLogger, DummyRun
import torch.nn as nn
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import CheckpointCallback
from datetime import datetime
import torch
import random


using_wandb = False

config = {
    "algo": "PPO",
    "total_timesteps": 500_000,
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

kwargs_PPO = {
    "policy": "MultiInputPolicy",
    "policy_kwargs": {
        "net_arch": {"pi": [16, 32, 64, 32, 16], "vf": [16, 32, 64, 32, 16]},
        "activation_fn": nn.ReLU,
        "ortho_init": True,
    },
    "verbose": 1,
}

# Get today's date
today = datetime.now()
formatted_date = today.strftime("%d-%m")


def make_env():
    env = gym.make(**env_config)
    env = Monitor(env)
    return env


env = SubprocVecEnv([make_env for i in range(config["num_envs"])], start_method="fork")

if using_wandb:
    run = wandb.init(
        project="swarm_env_t2",
        config=config,
        sync_tensorboard=True,
        monitor_gym=True,
        save_code=True,
    )
else:
    run = DummyRun()

env = VecVideoRecorder(
    env,
    f"videos/{formatted_date}/{run.id}",
    record_video_trigger=lambda x: x % 5_000 == 0,
    video_length=config["max_steps"],
)

episodic_callback = EpisodicRewardLogger(verbose=1)
wandbcallback = WandbCallback(
    gradient_save_freq=0,
    model_save_path=f"models/{formatted_date}/{run.id}",
    verbose=2,
)

# Save a checkpoint every 1000 steps
checkpoint_callback = CheckpointCallback(
    save_freq=5_000,
    save_path=f"./checkpoints/{formatted_date}/{run.id}",
    name_prefix=f"model_{run.id}",
    save_replay_buffer=True,
    save_vecnormalize=True,
)


algo_map = {"PPO": PPO, "SAC": SAC, "A2C": A2C, "R_PPO": RecurrentPPO}
model = algo_map[config["algo"]](
    env=env, tensorboard_log=f"runs/{formatted_date}/{run.id}", **kwargs_PPO
)

seed = 1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

model.learn(
    total_timesteps=config["total_timesteps"],
    callback=CallbackList([wandbcallback, episodic_callback, checkpoint_callback]),  # ,
    progress_bar=True,
)

print("ALGO ", config["algo"])
print(model.policy)

env.close()
run.finish()
