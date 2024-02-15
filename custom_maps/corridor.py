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
    RescueCenter,
    wounded_rescue_center_collision,
)
from spg_overlay.gui_map.closed_playground import ClosedPlayground
from spg_overlay.gui_map.map_abstract import MapAbstract
from custom_maps.wall_corridor import add_walls, add_boxes
from spg_overlay.reporting.evaluation import ZonesConfig
from spg_overlay.utils.misc_data import MiscData
from spg_overlay.entities.wounded_person import WoundedPerson


class Corridor(MapAbstract):
    def __init__(self, zones_config: ZonesConfig = ()):
        super().__init__(zones_config)
        self._time_step_limit = 2000
        self._real_time_limit = 120

        # PARAMETERS FOR MAP
        self._size_area = (800, 800)
        self._drones_pos = []
        # self._wounded_persons_pos = [(200, 0)]
        # self._number_wounded_persons = len(self._wounded_persons_pos)
        # self._wounded_persons: List[WoundedPerson] = []
        # self._rescue_center = RescueCenter(size=(200, 80))
        self._rescue_center_pos = ((-50, -50), 0)
        self._number_drones = 1

        # POSITION OF THE DRONE
        # Random orientation between -pi and pi
        orient = random.uniform(-math.pi, math.pi)
        drone_position = ((295, 118), orient)  # Example position and orientation
        self._drones_pos.append(drone_position)

    def construct_playground(self, drone_type: Type[DroneAbstract]) -> Playground:
        playground = ClosedPlayground(size=self._size_area)

        # Add walls to the playground
        add_boxes(playground)
        add_walls(playground)
        self._explored_map.initialize_walls(playground)

        # for i in range(self._number_wounded_persons):
        #     wounded_person = WoundedPerson(rescue_center=self._rescue_center)
        #     self._wounded_persons.append(wounded_person)
        #     pos = (self._wounded_persons_pos[i], 0)
            # playground.add(wounded_person, pos)

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
