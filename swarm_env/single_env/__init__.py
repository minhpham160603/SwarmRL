from gymnasium.envs.registration import register

register(
    id="SwarmEnv-v0",
    entry_point="swarm_env.single_env.single_agent:SwarmEnv",
)
