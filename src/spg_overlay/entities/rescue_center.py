from typing import Tuple
from arcade import Texture, load_texture

from spg.element import PhysicalElement
from spg.playground import Playground, get_colliding_entities
from spg.utils.definitions import CollisionTypes

from resources import path_resources
from spg_overlay.entities.wounded_person import WoundedPerson
import numpy as np


def wounded_rescue_center_collision(arbiter, _, data):
    playground: Playground = data["playground"]
    (wounded_person, _), (rescue_center, _) = get_colliding_entities(
        playground, arbiter
    )

    assert isinstance(wounded_person, WoundedPerson)
    assert isinstance(rescue_center, RescueCenter)

    if wounded_person.rescue_center == rescue_center:
        rescue_center.activate(wounded_person)

    return True


class RescueCenter(PhysicalElement):
    """
    The RescueCenter class represents a rescue center in the simulation. When a WoundedPerson comes into contact with
    the rescue center, it provides a reward of 1 to the drone bringing the wounded person. Then the wounded person
    disappears.
    """

    def __init__(self, size: Tuple[int, int], **kwargs):
        filename = path_resources + "/rescue_center.png"
        width = size[0]
        height = size[1]
        orig_x = int((800 - width) / 2)
        orig_y = int((800 - height) / 2)
        texture: Texture = load_texture(
            file_name=filename, x=orig_x, y=orig_y, width=width, height=height
        )
        super().__init__(texture=texture, **kwargs)

        self._quantity_rewards = None
        self._count_rewards = 0

    def _set_pm_collision_type(self):
        for pm_shape in self._pm_shapes:
            pm_shape.collision_type = CollisionTypes.ACTIVABLE_BY_GEM

    def activate(self, entity: WoundedPerson):
        if self._playground is None:
            raise ValueError("RescueCenter is not associated with a playground.")

        grasped_by_list = entity.grasped_by.copy()
        agents = [x.agent for x in grasped_by_list]
        if len(agents) > 0:
            agent = self.get_closest_agent(agents)
            for a in agents:
                a.base.grasper.reset()
        else:
            agent = self.get_closest_agent(self._playground.agents)
        agent.reward += 1
        self._playground.remove(entity)

    def get_closest_agent(self, agent_list):
        agent_pos = [agent.true_position() for agent in agent_list]
        center_pos = np.array(self.initial_coordinates[0])
        dist = [np.sqrt(np.sum((pos - center_pos) ** 2)) for pos in agent_pos]
        idx = np.argmin(dist)
        return agent_list[idx]
