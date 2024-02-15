import gymnasium as gym
import sys

from swarm_env.single_env.single_agent import SwarmEnv

env = SwarmEnv(
    render_mode="human",
    max_steps=100,
    fixed_step=20,
    map_name="MyMapIntermediate01",
)
# model = PPO.load(path) if path else PPO(env=env, policy="MultiInputPolicy")
for i in range(10):
    # print(model.policy)
    # raise
    obs, info = env.reset()
    # print(info)
    score = 0
    count = 0
    while True:
        # action, _states = model.predict(obs)
        action = env.action_space.sample()
        obs, reward, ter, trunc, info = env.step(action)
        count += 1
        score += reward
        if trunc or ter:
            print(f"Truc {trunc}, ter: {ter}, return: {score}, steps: {count}")
            break
    env.close()
