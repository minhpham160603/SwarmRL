import math
import random
from typing import List, Type

from spg.playground import Playground
from spg.utils.definitions import CollisionTypes

from spg_overlay.entities.drone_abstract import (
    DroneAbstract,
    drone_collision_wall,
    drone_collision_drone,
)
from spg_overlay.entities.rescue_center import (
    wounded_rescue_center_collision,
)

from spg_overlay.entities.wounded_person import WoundedPerson
from spg_overlay.gui_map.closed_playground import ClosedPlayground
from spg_overlay.gui_map.map_abstract import MapAbstract
from spg_overlay.reporting.evaluation import ZonesConfig
from spg_overlay.utils.misc_data import MiscData
import numpy as np


class BaseRLMap(MapAbstract):
    """
    Base class for the map that runs the RL env.
    Need to define:
    - self._drones: List
    - self._size_area: (width, height)
    - self._rescue_center_pos and self._rescue_center
    - self._wounded_persons_pos and _wounded_persons
    """

    def __init__(self, zones_config: ZonesConfig = ()):
        super().__init__(zones_config)

    def generate_position_v1(self):
        return (
            np.random.randint(
                -self._size_area[0] // 2 + 30, self._size_area[0] // 2 - 30
            ),
            np.random.randint(
                -self._size_area[1] // 2 + 30, self._size_area[1] // 2 - 30
            ),
        )

    def reset_drone(self):
        for i in range(self._number_drones):
            pos = self.generate_position_v1()
            angle = random.uniform(-math.pi, math.pi)
            self._drones[i].initial_coordinates = (pos, angle)

    def get_distance(self, pos_a, pos_b):
        return np.sqrt((pos_a[0] - pos_b[0]) ** 2 + (pos_a[1] - pos_b[1]) ** 2)

    def reset_rescue_center(self):
        self._rescue_center_pos = (
            self.generate_position_v1(),
            0,
        )
        self._rescue_center.initial_coordinates = self._rescue_center_pos
        self._rescue_center.move_to(self._rescue_center_pos)

    def reset_wounded_person(self):
        for i in range(self.number_wounded_persons):
            new_position = self.generate_position_v1()
            while self.get_distance(new_position, self._rescue_center_pos[0]) <= 120:
                new_position = self.generate_position_v1()
            pos = (new_position, 0)
            self._wounded_persons_pos[i] = pos
            self._wounded_persons[i].initial_coordinates = pos

    def add_wall_and_box(self, playground):
        pass

    def construct_playground(self, drone_type: Type[DroneAbstract]) -> Playground:
        playground = ClosedPlayground(size=self._size_area)
        # RESCUE CENTER
        playground.add_interaction(
            CollisionTypes.GEM,
            CollisionTypes.ACTIVABLE_BY_GEM,
            wounded_rescue_center_collision,
        )
        playground.add(self._rescue_center, self._rescue_center_pos)

        self.add_wall_and_box(playground)
        self._explored_map.initialize_walls(playground)

        # POSITIONS OF THE WOUNDED PERSONS
        for i in range(self._number_wounded_persons):
            wounded_person = WoundedPerson(rescue_center=self._rescue_center)
            self._wounded_persons.append(wounded_person)
            pos = (self._wounded_persons_pos[i], 0)
            playground.add(wounded_person, pos)

        # POSITIONS OF THE DRONES
        misc_data = MiscData(
            size_area=self._size_area, number_drones=self._number_drones
        )
        for i in range(self._number_drones):
            drone = drone_type(identifier=i, misc_data=misc_data)
            self._drones.append(drone)
            playground.add(drone, self._drones_pos[i])

        playground.add_interaction(
            CollisionTypes.PART, CollisionTypes.ELEMENT, drone_collision_wall
        )
        playground.add_interaction(
            CollisionTypes.PART, CollisionTypes.PART, drone_collision_drone
        )

        return playground
