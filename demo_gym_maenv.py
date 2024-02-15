from swarm_env.multi_env.multi_agent_gym import MultiSwarmEnv

env = MultiSwarmEnv(render_mode="human", n_agents=3, n_targets=3)
obs, info = env.reset()

while True:
    actions = env.sample_action()
    obs, reward, trunc, term, info = env.step(actions)
    if any(trunc) or any(term):
        break

env.close()
