from swarm_env.multi_env.multi_agent_comm import MultiSwarmEnv

env = MultiSwarmEnv(render_mode="human", n_agents=3, n_targets=3)
obs = env.reset()

while True:
    actions = env.sample_action()
    obs, reward, dones, info = env.step(actions)
    if any(dones):
        break

env.close()
