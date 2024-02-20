import math
import random
from typing import List, Type

from spg.playground import Playground
from spg.utils.definitions import CollisionTypes

from spg_overlay.entities.drone_abstract import DroneAbstract, drone_collision_wall, drone_collision_drone
from spg_overlay.entities.rescue_center import RescueCenter, wounded_rescue_center_collision
from spg_overlay.entities.sensor_disablers import ZoneType, NoComZone, NoGpsZone, KillZone, \
    srdisabler_disables_device
from spg_overlay.entities.wounded_person import WoundedPerson
from spg_overlay.gui_map.closed_playground import ClosedPlayground
from spg_overlay.gui_map.map_abstract import MapAbstract
from spg_overlay.reporting.evaluation import ZonesConfig
from spg_overlay.utils.misc_data import MiscData

from custom_maps.wall_medium_1 import add_walls, add_boxes


class CustomMedium1(MapAbstract):

    def __init__(self, zones_config: ZonesConfig = ()):
        super().__init__(zones_config)
        self._time_step_limit = 7200
        self._real_time_limit = 720  # In seconds

        # PARAMETERS MAP
        self._size_area = (1200, 900)
        self._number_drones = 1
        start_area_drones = (-200, -400)
        nb_per_side = math.ceil(math.sqrt(float(self._number_drones)))
        dist_inter_drone = 40.0
        # print("nb_per_side", nb_per_side)
        # print("dist_inter_drone", dist_inter_drone)
        sx = start_area_drones[0] - (nb_per_side - 1) * 0.5 * dist_inter_drone
        sy = start_area_drones[1] - (nb_per_side - 1) * 0.5 * dist_inter_drone
        # print("sx", sx, "sy", sy)

        self._drones_pos = []
        for i in range(self._number_drones):
            x = sx + (float(i) % nb_per_side) * dist_inter_drone
            y = sy + math.floor(float(i) / nb_per_side) * dist_inter_drone
            angle = random.uniform(-math.pi, math.pi)
            self._drones_pos.append(((x, y), angle))

        self._drones: List[DroneAbstract] = []

    def construct_playground(self, drone_type: Type[DroneAbstract]) -> Playground:
            playground = ClosedPlayground(size=self._size_area)

            add_walls(playground)
            add_boxes(playground)

            self._explored_map.initialize_walls(playground)

            # POSITIONS OF THE DRONES
            misc_data = MiscData(size_area=self._size_area,
                                number_drones=self._number_drones)
            for i in range(self._number_drones):
                drone = drone_type(identifier=i, misc_data=misc_data)
                self._drones.append(drone)
                playground.add(drone, self._drones_pos[i])

            playground.add_interaction(CollisionTypes.PART,
                                    CollisionTypes.ELEMENT,
                                    drone_collision_wall)
            playground.add_interaction(CollisionTypes.PART,
                                    CollisionTypes.PART,
                                    drone_collision_drone)

            return playground