from spg_overlay.entities.normal_wall import NormalWall, NormalBox

def add_boxes(playground):
    # box 0
    box = NormalBox(up_left_point=(-150, 150),
                    width=300, height=300)
    playground.add(box, box.wall_coordinates)


def add_walls(playground):
    # Assuming the map size is defined as an attribute of the class
    map_width, map_height = (800, 800)

    top_wall = NormalWall(
        pos_start=(-map_width / 2, map_height / 2),
        pos_end=(map_width / 2, map_height / 2),
    )
    playground.add(top_wall, top_wall.wall_coordinates)

    # Bottom wall
    bottom_wall = NormalWall(
        pos_start=(-map_width / 2, -map_height / 2),
        pos_end=(map_width / 2, -map_height / 2),
    )
    playground.add(bottom_wall, bottom_wall.wall_coordinates)

    # Left wall
    left_wall = NormalWall(
        pos_start=(-map_width / 2, -map_height / 2),
        pos_end=(-map_width / 2, map_height / 2),
    )
    playground.add(left_wall, left_wall.wall_coordinates)

    # Right wall
    right_wall = NormalWall(
        pos_start=(map_width / 2, -map_height / 2),
        pos_end=(map_width / 2, map_height / 2),
    )
    playground.add(right_wall, right_wall.wall_coordinates)

    # Inner rectangle closer to the outer walls
    inner_margin = 100  # Smaller margin for inner walls
    door_width = 80  # Width of the doors

    # Inner Top wall
    inner_top_wall_1 = NormalWall(
        pos_start=(-map_width / 2 + inner_margin, map_height / 2 - inner_margin),
        pos_end=(-door_width / 2, map_height / 2 - inner_margin),
    )
    inner_top_wall_2 = NormalWall(
        pos_start=(door_width / 2, map_height / 2 - inner_margin),
        pos_end=(map_width / 2 - inner_margin, map_height / 2 - inner_margin),
    )
    playground.add(inner_top_wall_1, inner_top_wall_1.wall_coordinates)
    playground.add(inner_top_wall_2, inner_top_wall_2.wall_coordinates)

    # Inner Bottom wall
    inner_bottom_wall_1 = NormalWall(
        pos_start=(-map_width / 2 + inner_margin, -map_height / 2 + inner_margin),
        pos_end=(-door_width / 2, -map_height / 2 + inner_margin),
    )
    inner_bottom_wall_2 = NormalWall(
        pos_start=(door_width / 2, -map_height / 2 + inner_margin),
        pos_end=(map_width / 2 - inner_margin, -map_height / 2 + inner_margin),
    )
    playground.add(inner_bottom_wall_1, inner_bottom_wall_1.wall_coordinates)
    playground.add(inner_bottom_wall_2, inner_bottom_wall_2.wall_coordinates)

    # Inner Left wall
    inner_left_wall = NormalWall(
        pos_start=(-map_width / 2 + inner_margin, -map_height / 2 + inner_margin),
        pos_end=(-map_width / 2 + inner_margin, map_height / 2 - inner_margin),
    )
    playground.add(inner_left_wall, inner_left_wall.wall_coordinates)

    # Inner Right wall
    inner_right_wall = NormalWall(
        pos_start=(map_width / 2 - inner_margin, -map_height / 2 + inner_margin),
        pos_end=(map_width / 2 - inner_margin, map_height / 2 - inner_margin),
    )
    playground.add(inner_right_wall, inner_right_wall.wall_coordinates)
