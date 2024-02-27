from gymnasium.spaces.space import Space
import numpy as np
from numpy import ndarray
import pygame
import gymnasium as gym
from gymnasium import spaces
import cv2
from swarm_env.env_renderer import GuiSR
from spg_overlay.entities.drone_distance_sensors import DroneSemanticSensor
from swarm_env.multi_env.ma_drone import MultiAgentDrone
import gc
from custom_maps.intermediate01 import MyMapIntermediate01
from custom_maps.easy import EasyMap
from typing import Any, Dict, Generic, Iterable, Iterator, TypeVar
from pettingzoo import ParallelEnv
from gymnasium.utils import EzPickle, seeding
from swarm_env.constants import *

ObsType = TypeVar("ObsType")
ActionType = TypeVar("ActionType")
AgentID = TypeVar("AgentID")


"""
Environment for multi agent
"""


map_dict = {
    "MyMapIntermediate01": MyMapIntermediate01,
    # "Corridor": Corridor,
    # "MultiRoom": MultiRoom,
    # "CustomMedium1": CustomMedium1,
    # "CustomMedium2": CustomMedium2,
    "Easy": EasyMap,
}


class MultiSwarmEnv(ParallelEnv, EzPickle):
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
            self._map = map_dict[map_name](num_drones=n_agents, num_persons=n_targets)
        else:
            raise Exception("Invalid map name")

        self.map_size = self._map.size_area
        self.continuous_action = continuous_action
        self.share_reward = share_reward
        self.n_agents = n_agents

        self._playground = self._map.construct_playground(drone_type=MultiAgentDrone)
        self._agents = self._playground._agents
        self.name_to_agent = {f"agent_{i}": a for i, a in enumerate(self._agents)}
        self.possible_agents = list(self.name_to_agent.keys())
        # Starting with this
        self.agents = (
            self.possible_agents
        )  # for now, agents don't die, so it remains similar
        self.fixed_step = fixed_step

        ### OBSERVATION
        self.observation_spaces = {
            agent_id: spaces.Dict(
                {
                    "lidar": spaces.Box(low=0, high=400, shape=(180,)),
                    # semantic: distance, angle, type: [0: nothing, 1: human, 2: rescue center]
                    "semantic": spaces.Box(
                        low=-np.inf,
                        high=np.inf,
                        shape=(1 + MAX_NUM_PERSONS + (MAX_NUM_DRONES - 1), 3),
                    ),
                    "pose": spaces.Box(low=-np.inf, high=np.inf, shape=(3,)),
                    "velocity": spaces.Box(low=-np.inf, high=np.inf, shape=(2,)),
                }
            )
            for agent_id in self.possible_agents
        }

        # forward, lateral, rotation, grasper
        self.action_spaces = {
            agent_id: spaces.Box(
                low=np.array([-1, -1, -1, 0]), high=np.array([1, 1, 1, 1]), shape=(4,)
            )
            for agent_id in self.possible_agents
        }

        self.current_rescue_count = 0
        self.current_step = 0
        self.max_episode_steps = max_episode_steps
        self.last_exp_score = None
        self.render_mode = render_mode
        self.gui = GuiSR(self._playground, self._map)
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
            "forward": action[0],
            "lateral": action[1],
            "rotation": action[2],
            "grasper": 1 if action[3] > 0.5 else 0,
        }

    def observe(self, agent_id):
        agent = self.name_to_agent[agent_id]
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
        semantic = np.zeros((1 + MAX_NUM_PERSONS + (MAX_NUM_DRONES - 1), 3)).astype(
            np.float32
        )
        center, human, drone = agent.process_special_semantic()

        semantic[0] = center[0]
        for i in range(min(len(human), MAX_NUM_PERSONS)):
            semantic[1 + i] = human[i]

        for i in range(min(len(drone), MAX_NUM_DRONES - 1)):
            semantic[1 + MAX_NUM_PERSONS + i] = drone[i]

        observation["semantic"] = semantic
        return observation

    def _get_obs(self):
        observations = {}
        for name in self.possible_agents:
            observations[name] = self.observe(agent_id=name)
        return observations

    def get_agent_info(self, agent_id):
        info = {}
        info["map_name"] = self.map_name
        info["wounded_people_pos"] = self._map._wounded_persons_pos
        info["rescue_zone"] = self._map._rescue_center_pos
        info["drones_true_pos"] = {
            agent_id: self.name_to_agent[agent_id].true_position()
        }
        return info

    def _get_info(self):
        infos = {}
        for agent_id in self.possible_agents:
            infos[agent_id] = self.get_agent_info(agent_id)
        return infos

    def reset_map(self):
        self._map.explored_map.reset()
        self._map.reset_rescue_center()
        self._map.reset_wounded_person()
        self._map.reset_drone()

    def reset(self, seed=None, options=None):
        # Reinit GUI
        gc.collect()
        self._playground.window.switch_to()
        self.reset_map()
        self._playground.reset()
        self.current_rescue_count = 0
        self.current_step = 0
        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def reward(self, agent, action):
        rew = -np.abs(action[2])
        if agent.is_collided():
            rew -= 1
        if agent.touch_human():
            rew += 1
        return rew

    def step(self, actions):
        self._playground.window.switch_to()
        frame_skip = 5
        counter = 0
        done = False
        steps = self.fixed_step  # 25 + int(action[4]) if self.fixed_step == 0 else
        prev_distances = [0] * self.n_agents
        for i, person in enumerate(self._map._wounded_persons):
            position = person.position
            prev_distances[i] = self.get_distance(
                (position[0], position[1]),
                self._map._rescue_center_pos[0],
            )

        # rotate value from -1 to 1, do this to discourage it from rotate to much

        commands = {}
        for agent_id in self.possible_agents:
            agent = self.name_to_agent[agent_id]
            move = self.construct_action(actions[agent_id])
            commands[agent] = move

        terminated, truncated = False, False
        rewards = {name: -0.5 for name in self.possible_agents}

        while counter < steps and not done:
            _, _, _, done = self._playground.step(commands)
            for name in self.possible_agents:
                agent = self.name_to_agent[name]
                if agent.reward != 0:
                    self.current_rescue_count += agent.reward
                    rewards[name] += 50

            if self.current_rescue_count >= self._map._number_wounded_persons:
                terminated = True
                break

            if self.render_mode == "human" and counter % frame_skip == 0:
                # self._agent.update_grid()
                self._render_frame()
            counter += 1

        for name in self.possible_agents:
            agent = self.name_to_agent[name]
            rewards[name] += self.reward(agent, actions[name])

        self.current_step += 1
        if self.current_step >= self.max_episode_steps:
            truncated = True
            for agent in rewards.keys():
                rewards[agent] -= 20

        # SHARED REWARD DEFINITION
        shared_reward = sum(rewards.values())

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
            final_rewards = {
                agent_id: shared_reward for agent_id in self.possible_agents
            }
        else:
            final_rewards = rewards
        terminations = {agent_id: terminated for agent_id in self.possible_agents}
        truncations = {agent_id: truncated for agent_id in self.possible_agents}

        observations = self._get_obs()
        infos = self._get_info()
        for agent_id in self.possible_agents:
            infos[agent_id]["individual_reward"] = rewards[agent_id]

        if self.render_mode == "human":
            self._render_frame()

        return observations, final_rewards, terminations, truncations, infos

    def _render_frame(self):
        # Capture the frame
        image = self.gui.get_playground_image()

        if self.render_mode == "human":
            for name in self.agents:
                color = (255, 0, 0)
                offset = 10
                agent = self.name_to_agent[name]
                pt1 = (
                    agent.true_position()
                    + np.array(self.map_size) / 2
                    + np.array([offset, offset])
                )
                org = (int(pt1[0]), self.map_size[1] - int(pt1[1]))
                str_id = name
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
        actions = {}
        for agent_id in self.possible_agents:
            actions[agent_id] = self.action_spaces[agent_id].sample()
        return actions

    def close(self):
        gc.collect()
        cv2.destroyAllWindows()

    def observation_space(self, agent: Any) -> Space:
        return self.observation_spaces[agent]

    def action_space(self, agent: Any) -> Space:
        return self.action_spaces[agent]


"""
if self.episode_count % self.swap_env_count == 0:
            if self.orig_map_name is None:
                selected_map = random.choice(list(map_dict.items()))
                self._map = selected_map[1]()
                self.map_name = selected_map[0]
            elif self.orig_map_name in map_dict: 
                self.map_name = self.orig_map_name
                self._map = map_dict[self.map_name]()
            assert self._map._number_drones == 1
            self._playground = self._map.construct_playground(drone_type=MidDrone)
            self._agent = self._playground._agents[0]
            self.gui = GuiSR(self._playground, self._map)
        else:
"""
