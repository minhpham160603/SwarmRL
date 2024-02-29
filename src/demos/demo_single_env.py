import gymnasium as gym
import sys
from stable_baselines3 import PPO
from swarm_env.single_env.single_agent import SwarmEnv
import arcade
from tqdm import tqdm
from memory_profiler import profile
from src.swarmrl.src.custom_maps.easy import EasyMap
from src.swarmrl.src.swarm_env.single_env.single_drone import SwarmDrone
import time


@profile
def test():
    pbar = tqdm(total=50)
    map = EasyMap()
    playground = map.construct_playground(SwarmDrone)
    drone = playground.agents[0]
    action = {drone: {"forward": 0.5}}
    for i in range(50):
        for i in range(2000):
            playground.step(action)
        playground.reset()
        pbar.update()
    arcade.close_window()


n_targets = 1
total_ep = 50


# @profile
def main():
    env = SwarmEnv(
        # render_mode="human",
        max_steps=100,
        fixed_step=20,
        n_targets=n_targets,
        map_name="Easy",
    )

    pbar = tqdm(total=total_ep)
    # path = "/users/eleves-b/2021/minh.pham/thesis/train-swarm/src/swarmrl/src/training/logs/28-02/6prfgo7u/best_model.zip"
    # model = PPO.load(path=path)

    for i in range(total_ep):
        obs, info = env.reset()
        score = 0
        count = 0
        while True:
            # action, _states = model.predict(obs)
            action = env.action_space.sample()
            obs, reward, ter, trunc, info = env.step(action)
            count += 1
            score += reward
            if trunc or ter:
                # print(f"Truc {trunc}, ter: {ter}, return: {score}, steps: {count}")
                break
        pbar.update()
    env.close()


main()
# test()
