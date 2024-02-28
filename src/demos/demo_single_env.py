import gymnasium as gym
import sys
from stable_baselines3 import PPO
from swarm_env.single_env.single_agent import SwarmEnv
import arcade
from tqdm import tqdm

n_targets = 3
total_ep = 10

env = SwarmEnv(
    render_mode="human",
    max_steps=100,
    fixed_step=20,
    n_targets=n_targets,
    map_name="Easy",
)

# pbar = tqdm(total=total_ep)
for i in range(total_ep):
    obs, info = env.reset()
    score = 0
    count = 0
    while True:
        # action, _states = model.predict(obs)
        action = env.action_space.sample()
        obs, reward, ter, trunc, info = env.step(action)
        print(obs["grasper"])
        count += 1
        score += reward
        if trunc or ter:
            print(f"Truc {trunc}, ter: {ter}, return: {score}, steps: {count}")
            break
    # pbar.update()
env.close()
