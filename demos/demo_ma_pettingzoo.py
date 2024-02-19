import sys

sys.path.append("../")

import gymnasium as gym
import swarm_env.single_env

# import supersuit as ss
from swarm_env.multi_env.multi_agent_pettingzoo import MultiSwarmEnv
import sys
import time


env = MultiSwarmEnv(render_mode="human", n_agents=3, n_targets=3, map_name="Easy")
model = None
for i in range(10):
    obs, info = env.reset()

    score = 0
    count = 0
    while True:
        actions = env.sample_action()
        if model:
            actions = {}
            for agent in env.possible_agents:
                single_obs = obs[agent]
                action, _ = model.predict(single_obs)
                actions[agent] = action
        obs, reward, ter, trunc, info = env.step(actions)
        count += 1
        score += reward["agent_0"]
        if trunc["agent_0"] or ter["agent_0"]:
            print(f"Truc {trunc}, ter: {ter}, return: {score}, steps: {count}")
            break
    env.close()
