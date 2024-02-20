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
    NoComZone,
    NoGpsZone,
    KillZone,
    srdisabler_disables_device,
)
from spg_overlay.entities.wounded_person import WoundedPerson
from spg_overlay.gui_map.closed_playground import ClosedPlayground
from spg_overlay.gui_map.map_abstract import MapAbstract
from spg_overlay.reporting.evaluation import ZonesConfig
from spg_overlay.utils.misc_data import MiscData

from custom_maps.wall_medium_1 import add_walls, add_boxes
import numpy as np
from .map_base_class import BaseRLMap


class EasyMap(BaseRLMap):
    def __init__(self, zones_config: ZonesConfig = (), num_drones=1, num_persons=1):
        super().__init__(zones_config)
        self._time_step_limit = 7200
        self._real_time_limit = 720  # In seconds

        # PARAMETERS MAP
        self._size_area = (300, 300)
        self._number_drones = num_drones
        self._drones: List[DroneAbstract] = []
        self._drones_pos = []

        for i in range(self._number_drones):
            x, y = self.generate_position_v1()
            angle = random.uniform(-math.pi, math.pi)
            self._drones_pos.append(((x, y), angle))

        self._number_wounded_persons = num_persons
        self._wounded_persons_pos = []
        for i in range(self._number_wounded_persons):
            self._wounded_persons_pos.append(self.generate_position_v1())

        self._wounded_persons: List[WoundedPerson] = []
        self._rescue_center = RescueCenter(size=(60, 60))
        self._rescue_center_pos = (
            self.generate_position_v1(),
            0,
        )
