import json
from tdw.scene_data.scene_bounds import SceneBounds

def shift(bounds, id = 0):
    eps = 1e-3
    y = bounds.top[1]
    z1 = bounds.back[-1]+eps
    z2 = bounds.front[-1]-eps
    x1 = bounds.left[0]+eps
    x2 = bounds.right[0]-eps
    from random import random
    z = z1 + (0.2 + 0.6 * (id % 2) + 0.1 * random()) * (z2 - z1)
    x = x1 + (0.2 + 0.6 * (id // 2 % 2) + 0.1 * random()) * (x2 - x1)
    return x, y, z

def belongs_to_which_room(x: float, z: float, scene_bounds: SceneBounds):
    for i, region in enumerate(scene_bounds.regions):
        if region.is_inside(x, z):
            return i
    return -1


with open("./dataset/room_types.json") as f:
    room_functionals = json.load(f)


def get_total_rooms(floorplan_scene: str) -> int:
    '''

    ### example：
    ```
    >>> get_total_rooms("2b")
    8
    ```
    '''
    return len(room_functionals[floorplan_scene[0]][0])
    
    
def get_room_functional_by_id(floorplan_scene: str, floorplan_layout: int, room_id: int) -> str:
    '''

    ### example：
    ```
    >>> get_room_functional_by_id("2b", 1_1, 1_1)
    Livingroom
    ```
    '''
    return room_functionals[floorplan_scene[0]][floorplan_layout][room_id]