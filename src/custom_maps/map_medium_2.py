import math
import random
from typing import List 
from spg_overlay.entities.wounded_person import WoundedPerson
from spg_overlay.reporting.evaluation import ZonesConfig
from maps.walls_medium_02 import add_walls, add_boxes
from .map_base_class import BaseRLMap
from spg_overlay.entities.rescue_center import RescueCenter


class CustomMedium2(BaseRLMap):
    def __init__(self, zones_config: ZonesConfig = (), num_persons=1, num_drones=1):
        super().__init__(zones_config)
        self._time_step_limit = 7200
        self._real_time_limit = 720  # In seconds

        # PARAMETERS MAP
        self._size_area = (1000, 750)
        self._number_drones = num_drones
        self._drones = []
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
        self._rescue_center = RescueCenter(size=(150, 80))
        self._rescue_center_pos = (
            (400, -300),
            0,
        )

    def reset_rescue_center(self):
        pass 

    def add_wall_and_box(self, playground):
        add_walls(playground)
        add_boxes(playground)
