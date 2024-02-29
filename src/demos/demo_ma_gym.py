from swarm_env.multi_env.multi_agent_gym import MultiSwarmEnv
from tqdm import tqdm
import gc
from memory_profiler import profile


def main():
    num_run = 50
    pbar = tqdm(total=num_run)

    env = MultiSwarmEnv(
        render_mode="rgb_arrray", n_agents=3, n_targets=3, max_episode_steps=100
    )

    for i in range(num_run):
        obs = env.reset()
        while True:
            actions = env.sample_action()
            obs, reward, dones, info = env.step(actions)
            if any(dones):
                # print("Done")
                break
        pbar.update(1)
    env.close()


main()
