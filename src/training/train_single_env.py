import sys

sys.path.append("../")
import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3 import PPO, SAC, A2C
import gymnasium as gym
from swarm_env.single_env.single_agent import SwarmEnv  # import to use with gym.make()
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import (
    VecVideoRecorder,
    SubprocVecEnv,
)
import numpy as np
from utils import DummyRun
import torch.nn as nn
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    StopTrainingOnRewardThreshold,
    CallbackList,
)
from datetime import datetime
import torch
import random


import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description="Your training script.")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    parser.add_argument("--use_wandb", action="store_true", default=False)
    parser.add_argument("--use_record", action="store_true", default=False)
    parser.add_argument("--total_steps", type=int, help="Total timestep to train")
    # Add other arguments as needed
    return parser.parse_args()


args = parse_arguments()

use_wandb = args.use_wandb
use_record = args.use_record
seed = args.seed

config = {
    "exp_name": "easy_3_target",
    "algo": "PPO",
    "total_timesteps": args.total_steps,
    "max_steps": 100,
    "num_envs": 4,
}

env_config = {
    "max_steps": 100,
    "map_name": "Easy",
    "continuous_action": True,
    "fixed_step": 20,
    "use_exp_map": True,
    "size_area": (350, 350),
    "n_targets": 3,
}

kwargs_policy = {
    "policy": "MultiInputPolicy",
    "stats_window_size": 100,
    "n_steps": 256,
    "ent_coef": 5e-4,
    "policy_kwargs": {
        "net_arch": {"pi": [64, 64], "vf": [64, 64]},
        "activation_fn": nn.ReLU,
        "ortho_init": True,
    },
    "verbose": 1,
}

wandb_config = {}
for x in [config, env_config, kwargs_policy]:
    for k, v in x.items():
        wandb_config[k] = v

# Get today's date
today = datetime.now()
formatted_date = today.strftime("%d-%m")


def make_env():
    env = SwarmEnv(**env_config)
    env = Monitor(env)
    return env


random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

env = SubprocVecEnv([make_env for i in range(config["num_envs"])], start_method="fork")


if use_wandb:
    run = wandb.init(
        project="single-env-v2",
        config=wandb_config,
        sync_tensorboard=True,
        monitor_gym=True,
        save_code=True,
        name=f"{wandb_config['exp_name']}_{config['algo']}_seed{seed}",
        group=config["algo"],
    )
else:
    run = DummyRun()

if use_record:
    env = VecVideoRecorder(
        env,
        f"videos/{formatted_date}/{run.id}",
        record_video_trigger=lambda x: x % 25_000 == 0,
        video_length=config["max_steps"],
    )

wandbcallback = (
    WandbCallback(
        gradient_save_freq=0,
        model_save_path=f"models/{config['algo']}/{formatted_date}/{run.id}",
        verbose=2,
    )
    if use_wandb
    else None
)

# Save a checkpoint every 1000 steps
checkpoint_callback = CheckpointCallback(
    save_freq=5_000,
    save_path=f"./checkpoints/{config['algo']}/{formatted_date}/{run.id}",
    name_prefix=f"model_{run.id}",
    save_replay_buffer=True,
    save_vecnormalize=True,
)

eval_callback = EvalCallback(
    env,
    best_model_save_path=f"./eval/{config['algo']}/{formatted_date}/{run.id}",
    log_path=f"./eval/{config['algo']}/{formatted_date}/{run.id}",
    eval_freq=5_000,
    n_eval_episodes=10,
    deterministic=True,
    render=False,
)

callbacklist = [eval_callback]
if use_wandb:
    callbacklist.append(wandbcallback)
callbacklist = CallbackList(callbacklist)

algo_map = {"PPO": PPO, "SAC": SAC, "A2C": A2C, "R_PPO": RecurrentPPO}
model = algo_map[config["algo"]](
    env=env,
    tensorboard_log=f"runs/{config['algo']}/{formatted_date}/{run.id}",
    **kwargs_policy,
)

print("ALGO ", config["algo"])
print(model.policy)

model.learn(
    total_timesteps=config["total_timesteps"],
    callback=callbacklist,
    progress_bar=True,
)

print("ALGO ", config["algo"])
print(model.policy)

env.close()
run.finish()
