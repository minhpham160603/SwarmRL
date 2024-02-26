import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces
import cv2
from swarm_env.env_renderer import GuiSR
from swarm_env.single_env.single_drone import SwarmDrone
import gc
from custom_maps.intermediate01 import MyMapIntermediate01
from custom_maps.corridor import Corridor
from custom_maps.multiple_rooms import MultiRoom
from custom_maps.easy import EasyMap
from swarm_env.constants import *
import arcade

"""
Environment for single agent
"""

map_dict = {
    "MyMapIntermediate01": MyMapIntermediate01,
    # "Corridor": Corridor,
    # "MultiRoom": MultiRoom,
    # "CustomMedium1": CustomMedium1,
    # "CustomMedium2": CustomMedium2,
    "Easy": EasyMap,
}


class SwarmEnv(gym.Env):
    """
    Variables:
    - max_steps: total timesteps to run before terminate the episdoe
    - fixed_steps: number of steps to step the playground per command produce by the agent
    - map_name: select the maps to run in.

    Oservation Space:
    - Pose: true_position and angle.
    - Velocity: velocity x and y axis.
    - Semantic: Rescue center, human, and drones. Data: distance, ray_angle, grased.
    - Lidar: 180 distance rays.

    Action Space: continuous or multi-discrete
    - Forward, Lateral, Rotation: [-1, 1]
    - Grasper: {0, 1}

    Reward function
    - Every step: -0.5
    - If hit the wall: -1
    - Touch the person: +1
    - Grasp each person back to rescue center: +50
    - Rotation penalty: abs(rotation_value) - to avoid the agent constantly rotating to prevent the wall
    - Exploration increase score

    Terminate when bring all the humans back to the rescue center.

    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        map_name="Easy",
        render_mode="rgb_array",
        max_steps=100,
        continuous_action=True,
        fixed_step=20,
        use_exp_map=False,
    ):
        if map_name in map_dict:
            self.map_name = map_name
            self._map = map_dict[map_name]()
        else:
            raise Exception("Invalid map name")

        self.agent_name = "agent_0"
        self.continuous_action = continuous_action
        assert self._map._number_drones == 1

        self._playground = None
        self._agent = None

        self.fixed_step = fixed_step
        self.total_rescued = 0
        self.map_size = None

        self.observation_space = spaces.Dict(
            {
                "lidar": spaces.Box(low=0, high=10, shape=(180,)),
                "semantic": spaces.Box(
                    low=-np.inf, high=np.inf, shape=((1 + MAX_NUM_PERSONS), 3)
                ),
                "pose": spaces.Box(low=-np.inf, high=np.inf, shape=(3,)),
                "velocity": spaces.Box(low=-np.inf, high=np.inf, shape=(2,)),
            }
        )

        if self.continuous_action:
            self.action_space = spaces.Box(
                low=np.array([-1, -1, -1, 0]), high=np.array([1, 1, 1, 1]), shape=(4,)
            )
        else:
            self.action_space = spaces.MultiDiscrete([3, 3, 3, 2])
        self.current_step = 0
        self.max_steps = max_steps
        self.last_exp_score = None
        self.use_exp_map = use_exp_map
        self.ep_count = 0

        self.render_mode = render_mode
        self.gui = None
        self.clock = None

    def construct_action(self, action):
        if self.continuous_action:
            return {
                "forward": np.clip(action[0], -1, 1),
                "lateral": np.clip(action[1], -1, 1),
                "rotation": np.clip(action[2], -1, 1),
                "grasper": 1 if action[3] > 0.5 else 0,
            }
        else:
            return {
                "forward": action[0] - 1,  # do this because sb3 does not work with -1
                "lateral": action[1] - 1,
                "rotation": action[2] - 1,
                "grasper": action[3],
            }

    def get_distance(self, pos_a, pos_b):
        return np.sqrt((pos_a[0] - pos_b[0]) ** 2 + (pos_a[1] - pos_b[1]) ** 2)

    def _get_obs(self):
        observation = {}
        observation["lidar"] = (
            self._agent.lidar_values()[:-1].astype(np.float32) / LIDAR_MAX_RANGE
        )

        observation["velocity"] = self._agent.measured_velocity().astype(np.float32)
        position = self._agent.true_position()
        normalized_position = (
            position[0] / self.map_size[0],
            position[1] / self.map_size[1],
        )

        observation["pose"] = np.concatenate(
            (normalized_position, [self._agent.true_angle()]), axis=0
        ).astype(np.float32)

        semantic = np.zeros((1 + MAX_NUM_PERSONS, 3)).astype(np.float32)
        data = self._agent.process_special_semantic()

        for i in range(min(len(data), len(semantic))):
            semantic[i] = data[i]

        observation["semantic"] = semantic
        return observation

    def _get_info(self):
        info = {}
        info["map_name"] = self.map_name
        info["wounded_people_pos"] = self._map._wounded_persons_pos
        info["rescue_zone"] = self._map._rescue_center_pos
        info["drones_true_pos"] = self._agent.true_position()
        return info

    def reset_map(self):
        self._map.explored_map.reset()
        self._map.reset_rescue_center()
        self._map.reset_wounded_person()
        self._map.reset_drone()

    def re_init(self):
        self._map = self._map = map_dict[self.map_name]()
        self.map_size = self._map._size_area

        self._playground = self._map.construct_playground(drone_type=SwarmDrone)
        self._agent = self._playground._agents[0]
        self.gui = GuiSR(self._playground, self._map)

    def reset(self, seed=None, options=None):
        if self.ep_count == 0:
            arcade.close_window()
            del self._map
            del self._agent
            del self._playground
            del self.gui
            self.re_init()
        self.ep_count += 1
        gc.collect()
        self._playground.window.switch_to()
        self.reset_map()
        self._playground.reset()
        self.current_step = 0
        self.total_rescued = 0
        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def step(self, action):
        self._playground.window.switch_to()

        frame_skip = 5
        counter = 0
        done = False

        person_position = self._map._wounded_persons[0].position
        previous_dist = self.get_distance(
            (person_position[0], person_position[1]), self._map._rescue_center_pos[0]
        )

        terminated, truncated = False, False

        # ROTATION PENALTY
        reward = -0.5 - np.abs(
            action[2]
        )  # to discourage the drone from rotate too much

        while counter < self.fixed_step and not done:
            cmd = {self._agent: self.construct_action(action)}
            _, _, _, done = self._playground.step(cmd)
            if (
                self._agent.reward != 0
            ):  # Warning: reward = 0 if the agent do not grasp the person when bring it back to rescue center
                self.total_rescued += self._agent.reward
            if self.total_rescued == self._map._number_wounded_persons:
                reward += 50
                terminated = True
                break
            if self.render_mode == "human" and counter % frame_skip == 0:
                self._render_frame()
            counter += 1

        reward = reward - self._agent.is_collided() + self._agent.touch_human()

        person_position = self._map._wounded_persons[0].position
        current_dist = self.get_distance(
            (person_position[0], person_position[1]), self._map._rescue_center_pos[0]
        )

        # REWARDED WHEN MOVE PERSON CLOSER TO RECUE CENTER
        reward -= (current_dist - previous_dist) / 5
        self.current_step += 1

        # REWARDED WHEN EXPLORE MORE
        if self.use_exp_map:
            current_exp_score = self._map.explored_map.score()
            if self.last_exp_score is not None:
                delta_score = current_exp_score - self.last_exp_score
            else:
                delta_score = 0
            reward += 50 * delta_score

            self.last_exp_score = current_exp_score
            self.gui.update_explore_map()

        if self.current_step >= self.max_steps:
            truncated = True
            reward -= 20

        observation = self._get_obs()
        info = self._get_info()
        info["reward"] = reward
        info["done"] = truncated or terminated

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, truncated, info

    def _render_frame(self):
        image = self.gui.get_playground_image()

        if self.render_mode == "human":
            if self.clock is None:
                self.clock = pygame.time.Clock()
            cv2.imshow("Playground Image", image)
            cv2.waitKey(1)
            self.clock.tick(self.metadata["render_fps"])

        return image

    def close(self):
        gc.collect()
        cv2.destroyAllWindows()
        arcade.close_window()
