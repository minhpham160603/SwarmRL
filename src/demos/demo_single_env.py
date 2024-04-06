from stable_baselines3 import PPO, SAC, A2C, TD3
from sb3_contrib import RecurrentPPO
from swarm_env.single_env.single_agent import SwarmEnv
import imageio


def main():
    path = "../../models/single_agents/easy_3_target/SAC/jqhg5qky.zip"

    items = path.split("/")
    index = items.index("models")
    algo = items[index + 3]
    total_ep = 10

    env = SwarmEnv(
        render_mode="human",
        max_steps=100,
        fixed_step=20,
        n_targets=3,
        map_name="Easy",
        # size_area=(350, 350),
    )

    algo_map = {"PPO": PPO, "A2C": A2C, "SAC": SAC, "TD3": TD3, "R_PPO": RecurrentPPO}
    model = algo_map[algo].load(path=path)
    images = []

    for i in range(total_ep):
        obs, info = env.reset()
        score = 0
        count = 0
        while True:
            action, _states = model.predict(obs)
            obs, reward, ter, trunc, info = env.step(action)
            count += 1
            score += reward
            if trunc or ter:
                print(f"Truc {trunc}, ter: {ter}, return: {score}, steps: {count}")
                images.extend(info["ep_frames"])
                break
    imageio.mimsave("./demo.gif", images, fps=25)
    env.close()


if __name__ == "__main__":
    main()
