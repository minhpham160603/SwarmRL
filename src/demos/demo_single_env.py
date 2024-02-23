import gymnasium as gym
import sys
from stable_baselines3 import PPO
from swarm_env.single_env.single_agent import SwarmEnv

env = SwarmEnv(
    render_mode="rgb_array",
    max_steps=100,
    fixed_step=20,
    map_name="Easy",
)
path = "../../models/single_agents/51cyttcw/model.zip"
model = PPO.load(path) if path else PPO(env=env, policy="MultiInputPolicy")
for i in range(10):
    obs, info = env.reset()
    score = 0
    count = 0
    while True:
        action, _states = model.predict(obs)
        # action = env.action_space.sample()
        obs, reward, ter, trunc, info = env.step(action)
        # print(obs["semantic"])
        # print(info["drones_true_pos"], info["rescue_zone"])
        # dist = env.get_distance(info["drones_true_pos"], info["rescue_zone"][0])
        # dist_human = env.get_distance(
        #     info["drones_true_pos"], info["wounded_people_pos"][0][0]
        # )
        # print("Dist ", dist, " human ", dist_human)
        # input()
        count += 1
        score += reward
        if trunc or ter:
            print(f"Truc {trunc}, ter: {ter}, return: {score}, steps: {count}")
            break
    env.close()
