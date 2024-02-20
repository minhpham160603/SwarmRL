import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3 import PPO

class EpisodicRewardLogger(BaseCallback):
    def __init__(self, verbose=0):
        super(EpisodicRewardLogger, self).__init__(verbose)
        self.episodic_rewards = []
        self.episodic_lengths = []
        self.current_rewards = None
        self.current_lengths = None
        self.policy = None
        self.episodes = 0

    def _on_training_start(self) -> None:
        self.policy = self.locals["self"]
        num_envs = self.policy.n_envs
        # raise
        self.current_rewards = np.zeros(num_envs)
        self.current_lengths = np.zeros(num_envs)
        return super()._on_training_start()

    def _on_step(self) -> bool:
        # Get rewards and episode over info from environment
        rewards = self.locals['rewards']
        dones = self.locals['dones']

        # Update current rewards and lengths
        self.current_rewards += rewards
        self.current_lengths += 1

        # Check for episode completion
        for i, done in enumerate(dones):
            if done:
                self.episodes += 1
                self.episodic_rewards.append(self.current_rewards[i])
                self.episodic_lengths.append(self.current_lengths[i])
                self.logger.record(
                            "charts/episodic_return", self.current_rewards[i], self.episodes
                        )
                self.logger.record(
                    "charts/episodic_length", self.current_lengths[i], self.episodes
                )
                if self.verbose != 0:
                    self.logger.dump(self.policy.num_timesteps)
                self.current_rewards[i] = 0
                self.current_lengths[i] = 0

        return True
    

class DummyRun:
    def __init__(self) -> None:
        self.id = np.random.randint(2**10, 2**16)
    
    def finish(self):
        print("Finish training")
