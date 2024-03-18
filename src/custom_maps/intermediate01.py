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
from spg_overlay.entities.sensor_disablers import (
    ZoneType,
    NoGpsZone,
    srdisabler_disables_device,
)
from spg_overlay.entities.wounded_person import WoundedPerson
from spg_overlay.gui_map.closed_playground import ClosedPlayground
from spg_overlay.gui_map.map_abstract import MapAbstract
from spg_overlay.reporting.evaluation import ZonesConfig
from spg_overlay.utils.misc_data import MiscData

from .walls_intermediate_map_1 import add_walls, add_boxes
import numpy as np
from .map_base_class import BaseRLMap


class MyMapIntermediate01(BaseRLMap):
    def __init__(
        self,
        zones_config: ZonesConfig = (),
        num_drones=1,
        num_persons=1,
        size_area=None,
    ):
        super().__init__(zones_config)
        self._time_step_limit = 2000
        self._real_time_limit = 120

        # PARAMETERS MAP
        self._size_area = (800, 500)
        self._rescue_center = RescueCenter(size=(200, 80))
        self._rescue_center_pos = ((295, 205), 0)

        self._no_gps_zone = NoGpsZone(size=(400, 500))
        self._no_gps_zone_pos = ((-190, 0), 0)

        self._wounded_persons_base = (-310, -180)
        self._wounded_persons_pos = []

        for i in range(num_persons):
            # self._wounded_persons_pos.append(self.generate_position_v1())
            self._wounded_persons_pos.append(self._wounded_persons_base)

        self._number_wounded_persons = len(self._wounded_persons_pos)
        self._wounded_persons: List[WoundedPerson] = []

        self._drones_pos = []
        for i in range(num_drones):
            orient = random.uniform(-math.pi, math.pi)
            position = self.generate_position_v1()
            self._drones_pos.append((position, orient))
        self._number_drones = len(self._drones_pos)
        self._drones: List[DroneAbstract] = []

    def reset_rescue_center(self):
        pass

    def add_wall_and_box(self, playground):
        add_walls(playground)
        add_boxes(playground)
