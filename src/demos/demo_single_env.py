import gymnasium as gym
import sys
from stable_baselines3 import PPO
from swarm_env.single_env.single_agent import SwarmEnv
from guppy import hpy
import arcade
from tqdm import tqdm


env = SwarmEnv(
    # render_mode="human",
    max_steps=100,
    fixed_step=20,
    map_name="Easy",
)
path = "../../models/single_agents/51cyttcw/model.zip"
model = PPO.load(path) if path else PPO(env=env, policy="MultiInputPolicy")

for i in range(50):
    obs, info = env.reset()
    score = 0
    count = 0
    while True:
        # action, _states = model.predict(obs)
        action = env.action_space.sample()
        obs, reward, ter, trunc, info = env.step(action)
        # print(obs["semantic"])
        count += 1
        score += reward
        if trunc or ter:
            print(f"Truc {trunc}, ter: {ter}, return: {score}, steps: {count}")
            break

env.close()
