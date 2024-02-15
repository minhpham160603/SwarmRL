import swarm_env.single_env
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym
from swarm_env.multi_env.multi_agent import MultiSwarmEnv
import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    VecVideoRecorder,
    SubprocVecEnv,
)

env_config = {
    "map_name": "Easy",
    "max_episode_steps": 100,
    "continuous_action": True,
    "n_agents": 1,
    "n_targets": 1,
}

def make_env():
    env = MultiSwarmEnv(**env_config)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    #envs = ss.concat_vec_envs_v1(env, 4, num_cpus=1, base_class="stable_baselines3")
    return env

env = SubprocVecEnv([make_env for i in range(4)], start_method="fork")
obs = env.reset()
print(obs)