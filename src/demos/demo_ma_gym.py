from swarm_env.multi_env.multi_agent_gym import MultiSwarmEnv


def main():
    num_run = 10
    env = MultiSwarmEnv(
        render_mode="rgb_array", n_agents=3, n_targets=3, max_episode_steps=100
    )

    for i in range(num_run):
        obs = env.reset()
        while True:
            actions = env.sample_action()
            obs, reward, dones, info = env.step(actions)
            if any(dones):
                print("Done")
                break
    env.close()


if __name__ == "__main__":
    main()
