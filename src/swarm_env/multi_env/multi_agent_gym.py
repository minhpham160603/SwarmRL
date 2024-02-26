from gymnasium.spaces.space import Space
import numpy as np
from numpy import ndarray
import pygame
import gymnasium as gym
from gymnasium import spaces
import cv2
from swarm_env.env_renderer import GuiSR
from swarm_env.multi_env.ma_drone import MultiAgentDrone
import gc
from custom_maps.intermediate01 import MyMapIntermediate01
from custom_maps.easy import EasyMap
from typing import Any, Dict, Generic, Iterable, Iterator, TypeVar
from pettingzoo import ParallelEnv
from gymnasium.utils import EzPickle, seeding
from swarm_env.constants import *
import arcade

"""
Environment for multi agent
"""


map_dict = {
    "MyMapIntermediate01": MyMapIntermediate01,
    "Easy": EasyMap,
}


class MultiSwarmEnv(gym.Env):
    """
    Oservation
    GPS Position: 2
    Velocity: 2
    Humans: (distance, angle, grasped)
    Rescue zone: (distance, angle, grasped)
    Lidar: 180


    Reward
    - Every step: -0.05
    - If hit the wall: -5
    - If touch the person: +50
    - Entropy score # may not be neccesary as we have the delta exploration score
    - Exploration increase score

    Terminate when reach the wounded person

    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        map_name="Easy",
        n_agents=1,
        n_targets=1,
        render_mode="rgb_array",
        max_episode_steps=100,
        continuous_action=True,
        fixed_step=20,
        share_reward=True,
        use_exp_map=False,
    ):
        EzPickle.__init__(
            self,
            map_name=map_name,
            n_agents=n_agents,
            n_targets=n_targets,
            render_mode=render_mode,
            max_episode_steps=max_episode_steps,
            continuous_action=continuous_action,
            fixed_step=fixed_step,
        )

        if map_name in map_dict:
            self.map_name = map_name
        else:
            raise Exception("Invalid map name")

        self._map = None
        self.map_size = None
        self.continuous_action = continuous_action
        self.share_reward = share_reward
        self.n_agents = n_agents
        self.n_targets = n_targets

        self._playground = None
        self._agents = None
        self.agents = None

        self.fixed_step = fixed_step
        self.use_exp_map = use_exp_map

        ### OBSERVATION

        """
        Lidar: 180 + semantic: (1 + 3 + 2) * 3 + pose: 3 + velocity: 2 = 203
        """
        single_action_dim = 180 + (self.n_targets + self.n_agents) * 3 + 5
        self.observation_space = [
            spaces.Box(low=-np.inf, high=np.inf, shape=(single_action_dim,))
            for _ in range(self.n_agents)
        ]
        self.share_observation_space = [
            spaces.Box(
                low=-np.inf,
                high=+np.inf,
                shape=(single_action_dim * self.n_agents,),
                dtype=np.float32,
            )
            for _ in range(self.n_agents)
        ]

        # forward, lateral, rotation, grasper
        self.action_space = [
            spaces.Box(
                low=np.array([-1, -1, -1, 0]), high=np.array([1, 1, 1, 1]), shape=(4,)
            )
            for _ in range(self.n_agents)
        ]

        self.current_rescue_count = 0
        self.current_step = 0
        self.ep_count = 0
        self.max_episode_steps = max_episode_steps
        self.last_exp_score = None
        self.render_mode = render_mode
        self.gui = None
        self.clock = None

    def get_distance(self, pos_a, pos_b):
        return np.sqrt((pos_a[0] - pos_b[0]) ** 2 + (pos_a[1] - pos_b[1]) ** 2)

    def flatten_obs(self, observation):
        obs_array = np.array([])
        for v in observation.values():
            obs_array = np.concatenate((obs_array, v.flatten()), axis=0)
        return obs_array

    def state(self) -> ndarray:
        observations = self._get_obs()
        state_array = []
        for agent_id in self.agents:
            obs_array = self.flatten_obs(observations[agent_id])
            state_array.append(obs_array)
        return np.array(state_array)

    def construct_action(self, action):
        return {
            "forward": np.clip(action[0], -1, 1),
            "lateral": np.clip(action[1], -1, 1),
            "rotation": np.clip(action[2], -1, 1),
            "grasper": 1 if action[3] > 0.5 else 0,
        }

    def observe(self, agent_id):
        agent = self._agents[agent_id]
        observation = {}
        observation["lidar"] = (
            agent.lidar_values()[:-1].astype(np.float32) / LIDAR_MAX_RANGE
        )
        observation["velocity"] = agent.measured_velocity().astype(np.float32)
        normalized_position = (
            agent.true_position()[0] / self.map_size[0],
            agent.true_position()[1] / self.map_size[1],
        )
        observation["pose"] = np.concatenate(
            (normalized_position, [agent.true_angle()]), axis=0
        ).astype(np.float32)
        semantic = np.zeros((1 + self.n_targets + (self.n_agents - 1), 3)).astype(
            np.float32
        )
        center, human, drone = agent.process_special_semantic()

        semantic[0] = center[0]
        for i in range(min(len(human), self.n_targets)):
            semantic[1 + i] = human[i]

        for i in range(min(len(drone), self.n_agents - 1)):
            semantic[1 + self.n_targets + i] = drone[i]

        observation["semantic"] = semantic

        return self.flatten_obs(observation)

    def _get_obs(self):
        observations = []
        for idx in self.agents:
            observations.append(self.observe(agent_id=idx))
        return observations

    def seed(self, seed=None):
        if seed is None:
            np.random.seed(1)
        else:
            np.random.seed(seed)

    def get_agent_info(self, agent_id):
        info = {}
        info["map_name"] = self.map_name
        info["wounded_people_pos"] = self._map._wounded_persons_pos
        info["rescue_zone"] = self._map._rescue_center_pos
        info["drones_true_pos"] = {agent_id: self._agents[agent_id].true_position()}
        return info

    def _get_info(self):
        infos = {}
        for agent_id in self.agents:
            infos[agent_id] = self.get_agent_info(agent_id)
        return infos

    def reset_map(self):
        self._map.explored_map.reset()
        self._map.reset_rescue_center()
        self._map.reset_wounded_person()
        self._map.reset_drone()

    def re_init(self):
        self._map = map_dict[self.map_name](
            num_drones=self.n_agents, num_persons=self.n_targets
        )
        self.map_size = self._map._size_area

        self._playground = self._map.construct_playground(drone_type=MultiAgentDrone)
        self._agents = self._map.drones
        self.agents = [i for i in range(self.n_agents)]
        self.gui = GuiSR(self._playground, self._map)

    def reset(self, seed=None, options=None):
        if self.ep_count % 30 == 0:
            arcade.close_window()
            del self._map
            del self._agents
            del self._playground
            del self.gui
            self.re_init()
        self.ep_count += 1
        # Reinit GUI
        gc.collect()
        self._playground.window.switch_to()
        self.reset_map()
        self._playground.reset()

        self.current_step = 0
        observation = self._get_obs()
        info = self._get_info()
        return observation

    def render(self, mode=None):
        if mode:
            self.render_mode = mode
        return self._render_frame()

    def get_map(self):
        return self._map

    def reward(self, agent, action):
        rew = -np.abs(action[2])
        if agent.is_collided():
            rew -= 1
        if agent.touch_human():
            rew += 1
        for human in self._map._wounded_persons:
            magnets = set(human.grasped_by)
            if len(magnets) > 1 and agent.base.grasper in magnets:
                rew -= 1
                # print("Conflict!!")
        return rew

    def step(self, actions):
        self._playground.window.switch_to()
        frame_skip = 5
        counter = 0
        done = False
        steps = self.fixed_step
        prev_distances = [0] * self.n_agents
        for i, person in enumerate(self._map._wounded_persons):
            position = person.position
            prev_distances[i] = self.get_distance(
                (position[0], position[1]),
                self._map._rescue_center_pos[0],
            )

        # rotate value from -1 to 1, do this to discourage it from rotate to much

        commands = {}
        for i, agent in enumerate(self._agents):
            move = self.construct_action(actions[i])
            commands[agent] = move

        terminated, truncated = False, False
        rewards = [-0.5 for _ in range(self.n_agents)]

        while counter < steps and not done:
            _, _, _, done = self._playground.step(commands)

            for i, agent in enumerate(self._agents):
                if agent.reward != 0:
                    self.current_rescue_count += agent.reward
                    rewards[i] += 50

            if self.current_rescue_count >= self._map._number_wounded_persons:
                terminated = True
                self.current_rescue_count = 0
                break

            if self.render_mode == "human" and counter % frame_skip == 0:
                # self._agent.update_grid()
                self._render_frame()
            counter += 1

        for i, agent in enumerate(self._agents):
            rewards[i] += self.reward(agent, actions[i])

        self.current_step += 1
        if self.current_step >= self.max_episode_steps:
            truncated = True
            for i in range(len(rewards)):
                rewards[i] -= 20

        # SHARED REWARD DEFINITION
        shared_reward = sum(rewards)

        delta_distances = 0
        for i, person in enumerate(self._map._wounded_persons):
            position = person.position
            delta_distances += (
                self.get_distance(
                    (position[0], position[1]),
                    self._map._rescue_center_pos[0],
                )
                - prev_distances[i]
            )

        shared_reward -= delta_distances / 5

        if self.use_exp_map:
            current_exp_score = self._map.explored_map.score()
            if self.last_exp_score is not None:
                delta_exp_score = current_exp_score - self.last_exp_score
            else:
                delta_exp_score = 0

            self.last_exp_score = current_exp_score
            # print(f"score {delta_exp_score}, {current_exp_score}")
            self.gui.update_explore_map()

            # REWARD
            shared_reward += 50 * delta_exp_score

        if self.share_reward:
            final_rewards = [[shared_reward]] * self.n_agents
        else:
            final_rewards = rewards
        # terminations = [terminated] * self.n_agents
        # truncations = [truncated] * self.n_agents
        dones = [terminated or truncated] * self.n_agents

        observations = self._get_obs()
        infos = self._get_info()
        for i, agent_id in enumerate(self.agents):
            infos[agent_id]["individual_reward"] = rewards[i]

        if self.render_mode == "human":
            self._render_frame()

        return observations, final_rewards, dones, infos

    def _render_frame(self):
        # Capture the frame
        image = self.gui.get_playground_image()

        if self.render_mode == "human":
            for name in self.agents:
                color = (255, 0, 0)
                offset = 10
                agent = self._agents[name]
                pt1 = (
                    agent.true_position()
                    + np.array(self.map_size) / 2
                    + np.array([offset, offset])
                )
                org = (int(pt1[0]), self.map_size[1] - int(pt1[1]))
                str_id = str(name)
                font = cv2.FONT_HERSHEY_SIMPLEX
                image = cv2.putText(
                    image,
                    str_id,
                    org,
                    fontFace=font,
                    fontScale=0.4,
                    color=color,
                    thickness=1,
                )
            if self.clock is None:
                self.clock = pygame.time.Clock()
            cv2.imshow("Playground Image", image)
            cv2.waitKey(1)
            self.clock.tick(self.metadata["render_fps"])

        return image

    def sample_action(self):
        actions = []
        for i in range(self.n_agents):
            actions.append(self.action_space[i].sample())
        return actions

    def close(self):
        gc.collect()
        cv2.destroyAllWindows()
        arcade.close_window()

    def observation_space(self, agent: Any) -> Space:
        return self.observation_spaces[agent]

    def action_space(self, agent: Any) -> Space:
        return self.action_spaces[agent]
