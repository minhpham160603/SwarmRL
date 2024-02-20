# -*- coding: utf-8 -*-

# import random
from typing import Optional, Dict, Any, Tuple
import numpy as np
from spg_overlay.entities.drone_abstract import DroneAbstract
from spg_overlay.utils.misc_data import MiscData

from spg_overlay.utils.constants import (
    RESOLUTION_SEMANTIC_SENSOR,
    MAX_RANGE_SEMANTIC_SENSOR,
    FOV_SEMANTIC_SENSOR,
    RESOLUTION_LIDAR_SENSOR,
    FOV_LIDAR_SENSOR,
    MAX_RANGE_LIDAR_SENSOR,
)
from swarm_env import constants


class Color:
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    RESET = "\033[0m"


class SwarmDrone(DroneAbstract):
    """
    Very simple, light-weight drone to train SwarmEnv
    """

    def __init__(
        self,
        identifier: Optional[int] = None,
        misc_data: Optional[MiscData] = None,
        **kwargs,
    ):
        super().__init__(
            identifier=identifier,
            misc_data=misc_data,
            display_lidar_graph=False,
            **kwargs,
        )

        self.size_area = misc_data.size_area if misc_data else None
        self.iteration: int = 0
        self.droneReady = False  # changes to True when all drones are ready to go (and send normal messages)

        # Com
        self.msg_data = None
        # ray angles of the sensors (constant)
        self.lidar_ray_angles = self.lidar().ray_angles

        self.rescue_center_pos = None
        self.init_position = None

        # Logging
        self.show_log_info = False
        self.show_log_warning = False
        self.show_log_error = False
        self.semantic_data = None

        self.log_info("Drone is initialized")

    def define_message_for_all(self):
        # msg_data before start is assigned in the init state
        if not self.droneReady:
            return None

        return self.msg_data

    def is_collided(self):
        """
        Returns True if the drone collided a wall or other drones
        """
        if self.lidar_values() is None or self.semantic_values() is None:
            return False

        collided = False

        dist = min(self.lidar_values())
        semantic = self.semantic_values()
        if len(semantic) == 0:
            return dist < 30

        idx = np.argmin([x.distance for x in semantic])

        if dist < 30 and str(semantic[idx].entity_type) != "TypeEntity.WOUNDED_PERSON":
            collided = True
        return collided

    def touch_human(self):
        semantic = self.semantic_values()

        if semantic is None:
            return False

        for x in semantic:
            if x.distance < 30 and str(x.entity_type) == "TypeEntity.WOUNDED_PERSON":
                return True

        return False

    def get_distance(self, pos_a, pos_b):
        return np.sqrt((pos_a[0] - pos_b[0]) ** 2 + (pos_a[1] - pos_b[1]) ** 2)

    def control(self) -> Dict[str, Any]:
        """
        Drone explore the map and rescue victims when detected
        Drone bring back victims to the base and travel back to a repositioning position
        """
        return None

    def process_special_semantic(self):
        """Special semantic that also gives the identifier of the entity
        to give one semantic data only per entity.
        """
        semantic = self.semantic().special_semantic()
        data = {}
        for k, v, _ in semantic:
            if k in data or str(v.entity_type) == "TypeEntity.WALL":
                continue
            data[k] = v

        center = []
        human = []
        for d in data.values():
            item = np.zeros((3,), dtype=np.float32)
            item[0] = d.distance / constants.LIDAR_MAX_RANGE
            item[1] = d.angle
            item[2] = d.grasped
            if str(d.entity_type) == "TypeEntity.RESCUE_CENTER":
                center.append(item)
            elif str(d.entity_type) == "TypeEntity.WOUNDED_PERSON":
                human.append(item)
        if len(center) == 0:
            center = [np.zeros((3,))]
        human = sorted(human, key=lambda i: i[0])
        return center + human

    def log_info(self, log=""):
        if self.show_log_info:
            print(
                f"{Color.BOLD}{Color.WHITE}[{str(self.iteration).zfill(5)}][Drone {self.identifier}] {log}{Color.RESET}"
            )

    def log_warning(self, log=""):
        if self.show_log_warning:
            print(
                f"{Color.BOLD}{Color.YELLOW}[{str(self.iteration).zfill(5)}][Drone {self.identifier}] {log}{Color.RESET}"
            )

    def log_error(self, log=""):
        if self.show_log_error:
            print(
                f"{Color.BOLD}{Color.RED}[{str(self.iteration).zfill(5)}][Drone {self.identifier}] {log}{Color.RESET}"
            )
