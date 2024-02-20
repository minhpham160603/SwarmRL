from spg_overlay.entities.normal_wall import NormalWall, NormalBox

door_width = 80

def create_wall_with_door(start, end, is_horizontal):
    if is_horizontal:
        mid_point = (start[0] + end[0]) / 2
        return [(start, (mid_point - door_width / 2, start[1])), ((mid_point + door_width / 2, start[1]), end)]
    else:
        mid_point = (start[1] + end[1]) / 2
        return [(start, (start[0], mid_point - door_width / 2)), ((start[0], mid_point + door_width / 2), end)]

def add_walls(playground):
    # Walls without doors
    walls = [
        # Outer walls
        ((-400, -150), (-150, -150)),
        ((-150, -150), (-150, -400)),
        ((150, -400), (150, -150)),
        ((400, -150), (150, -150)),

        # Middle structure
        ((-150, 150), (150, 150)),
        ((-150, -150), (-150, 150)),
        ((150, -150), (150, 150)),
        ((-150, -150), (150, -150)),

        ((-400, 150), (-150, 150)),
        ((-150, 400), (-150, 150)),
        ((150, 150), (150, 400)),
        ((400, 150), (150, 150)),
    ]

    # Create walls with doors
    walls_with_doors = []
    for start, end in walls:
        is_horizontal = start[1] == end[1]
        walls_with_doors.extend(create_wall_with_door(start, end, is_horizontal))

    # Create and add walls to the playground
    for start, end in walls_with_doors:
        wall = NormalWall(pos_start=start, pos_end=end)
        playground.add(wall, wall.wall_coordinates)