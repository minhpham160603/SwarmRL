from swarm_env.multi_env.multi_agent_gym import MultiSwarmEnv

env = MultiSwarmEnv(
    render_mode="rgb_array", n_agents=3, n_targets=3, max_episode_steps=30
)

for i in range(50):
    obs = env.reset()
    while True:
        actions = env.sample_action()
        obs, reward, dones, info = env.step(actions)
        if any(dones):
            print("Done")
            break
env.close()
