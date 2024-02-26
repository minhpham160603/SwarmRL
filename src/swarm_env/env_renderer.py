import arcade
import time
from typing import Optional, Tuple, List, Dict, Union, Type
import cv2

from spg.agent.controller.controller import Command, Controller
from spg.playground import Playground
from spg.playground.playground import SentMessagesDict
from spg.view import TopDownView

from spg_overlay.utils.constants import FRAME_RATE, DRONE_INITIAL_HEALTH
from spg_overlay.entities.drone_abstract import DroneAbstract
from spg_overlay.entities.keyboard_controller import KeyboardController
from spg_overlay.utils.fps_display import FpsDisplay
from spg_overlay.gui_map.map_abstract import MapAbstract
from spg_overlay.utils.mouse_measure import MouseMeasure
from spg_overlay.reporting.screen_recorder import ScreenRecorder
from spg_overlay.utils.visu_noises import VisuNoises


class GuiSR(TopDownView):
    """
    The GuiSR class is a subclass of TopDownView and provides a graphical user interface for the simulation. It handles
    the rendering of the playground, drones, and other visual elements, as well as user input and interaction.
    """

    def __init__(
        self,
        playground: Playground,
        the_map: MapAbstract,
        size: Optional[Tuple[int, int]] = None,
        center: Tuple[float, float] = (0, 0),
        zoom: float = 1,
        use_keyboard: bool = False,
        display_uid: bool = False,
        draw_transparent: bool = False,
        draw_interactive: bool = False,
        draw_zone: bool = True,
        draw_lidar_rays: bool = False,
        draw_semantic_rays: bool = False,
        draw_gps: bool = False,
        draw_com: bool = False,
        print_rewards: bool = False,
        print_messages: bool = False,
        use_mouse_measure: bool = False,
        enable_visu_noises: bool = False,
        filename_video_capture: str = None,
    ) -> None:
        super().__init__(
            playground,
            size,
            center,
            zoom,
            display_uid,
            draw_transparent,
            draw_interactive,
            draw_zone,
        )
        self._playground.window.set_size(*self._size)
        self._playground.window.set_visible(False)
        self._the_map = the_map
        self._drones = self._the_map.drones

    def run(self):
        self._playground.window.run()

    def update_explore_map(self):
        self._the_map.explored_map.update_drones(self._drones)
        self._the_map.explored_map._process_positions()

    def on_update(self):
        pass

    def get_playground_image(self):
        self.update()
        # The image should be flip and the color channel permuted
        image = cv2.flip(self.get_np_img(), 0)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def draw(self, force=False):
        pass

    @property
    def last_image(self):
        return self._last_image

    @property
    def percent_drones_destroyed(self):
        return self._percent_drones_destroyed

    @property
    def mean_drones_health(self):
        return self._mean_drones_health

    @property
    def elapsed_time(self):
        return self._elapsed_time

    @property
    def real_time_elapsed(self):
        return self._real_time_elapsed

    @property
    def rescued_number(self):
        return self._rescued_number

    @property
    def rescued_all_time_step(self):
        return self._rescued_all_time_step

    @property
    def real_time_limit_reached(self):
        return self._real_time_limit_reached
