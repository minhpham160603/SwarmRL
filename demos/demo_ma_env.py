import gymnasium as gym

# from stable_baselines3 import PPO
import gymnasium as gym
import sys

sys.path.append("../")

import swarm_env.single_env

# import supersuit as ss
from swarm_env.multi_env.multi_agent import MultiSwarmEnv
import sys
import time


# path = "/home/mip012/Documents/Code/swarm-marl/src/swarm_rescue/models/11-02/xidrhr3c/model.zip"
path = None
"""Rendering"""
if len(sys.argv) == 2:
    # print("Usage: script.py <remote_file_path>")
    # sys.exit(1)
    path = sys.argv[1]


env = MultiSwarmEnv(render_mode="human", n_agents=3, n_targets=3, map_name="Easy")
model = None
# if path:
# model = PPO.load(path)
for i in range(10):
    obs, info = env.reset()

    score = 0
    count = 0
    while True:
        # action = env.action_space.sample()

        # action, _states = model.predict(obs)
        actions = env.sample_action()
        if model:
            actions = {}
            for agent in env.possible_agents:
                single_obs = obs[agent]
                action, _ = model.predict(single_obs)
                actions[agent] = action
        obs, reward, ter, trunc, info = env.step(actions)
        # print(obs["agent_0"]["semantic"])
        count += 1
        score += reward["agent_0"]

        # input()
        if trunc["agent_0"] or ter["agent_0"]:
            print(f"Truc {trunc}, ter: {ter}, return: {score}, steps: {count}")
            break
    env.close()
