import numpy as np
from typing import Any, Tuple, Sequence
import os.path as osp
import h5py
import numpy as np
import cv2
import math
# import magnum as mn
import quaternion
from PIL import Image, ImageDraw
import os
action_mapping={
    0:"stop",1:"forward",2:"left",3:"right"
}
def get_contour_points(pos, size):
    x, y, o = pos
    pt1 = (int(x),
           int(y))
    # pt2 = (int(x + size / 1.5 * np.cos(o + np.pi * 4 / 3)),
    #        int(y + size / 1.5 * np.sin(o + np.pi * 4 / 3)))
    # pt3 = (int(x + size * np.cos(o)),
    #        int(y + size * np.sin(o)))
    # pt4 = (int(x + size / 1.5 * np.cos(o - np.pi * 4 / 3)),
    #        int(y + size / 1.5 * np.sin(o - np.pi * 4 / 3)))
    
    pt2 = (int(x + size / 1.5 * np.sin(o + np.pi * 4 / 3)),
           int(y + size / 1.5 * np.cos(o + np.pi * 4 / 3)))
    pt3 = (int(x + size * np.sin(o)),
           int(y + size * np.cos(o)))
    pt4 = (int(x + size / 1.5 * np.sin(o - np.pi * 4 / 3)),
           int(y + size / 1.5 * np.cos(o - np.pi * 4 / 3)))

    return np.array([pt1, pt2, pt3, pt4])

import math
 
def euler_from_quaternion(w,x,y,z):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    # t0 = +2.0 * (w * x + y * z)
    # t1 = +1.0 - 2.0 * (x * x + y * y)
    # roll_x = math.atan2(t0, t1)
    
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)
    
    # t3 = +2.0 * (w * z + x * y)
    # t4 = +1.0 - 2.0 * (y * y + z * z)
    # yaw_z = math.atan2(t3, t4)
    
    # return roll_x, pitch_y, yaw_z # in radians
    return pitch_y

# def draw_agent(aloc, arot, np_map):
#     arot = quaternion.from_float_array(np.array([arot[3], *arot[:3]]) )
#     # arot = quaternion.from_float_array(np.array(arot))
#     agent_forward = mn.Quaternion(arot.imag, arot.real).transform_vector(mn.Vector3(0, 0, -1.0))
#     agent_orientation = math.atan2(agent_forward[0], agent_forward[2])
#     agent_arrow = get_contour_points( (aloc[1], aloc[0], agent_orientation), size=15)
#     cv2.drawContours(np_map, [agent_arrow], 0, (0,0,255,255), -1)

def draw_agent(aloc, arot, np_map):
    arot_q = quaternion.from_float_array(np.array([arot[3], *arot[:3]]) )
    agent_forward = quaternion.rotate_vectors(arot_q, np.array([0,0,-1.]))
    agent_orientation = math.atan2(agent_forward[0], agent_forward[2])
    agent_arrow = get_contour_points( (aloc[1], aloc[0], -agent_orientation), size=15)
    cv2.drawContours(np_map, [agent_arrow], 0, (0,0,255,255), -1)

def draw_point(pil_img, x, y, point_size, color):
    drawer = ImageDraw.Draw(pil_img, 'RGBA')
    drawer.ellipse((x-point_size, y-point_size, x+point_size, y+point_size), fill=color)

def draw_path(np_map, gt_annt, grid_dimensions, upper_bound, lower_bound):
    locations = gt_annt['locations']
    if os.environ.get('DEBUG', False): 
        print('\033[92m'+'DEBUG mode:'+'\033[0m')
        print("GT Action sequence: ", [action_mapping[act] for act in gt_annt['actions']])

    for i in range(1, len(locations)):
        start_grid_pos = simloc2maploc(
            locations[i-1], grid_dimensions, upper_bound, lower_bound
        )
        end_grid_pos = simloc2maploc(
            locations[i], grid_dimensions, upper_bound, lower_bound
        )
        cv2.line(
            np_map,
            [start_grid_pos[1], start_grid_pos[0]], # use x,y coord order
            [end_grid_pos[1], end_grid_pos[0]],
            color=(0,128,128,255),
            thickness=2,
        )

def simloc2maploc(aloc, grid_dimensions, upper_bound, lower_bound):
    agent_grid_pos = to_grid(
        aloc[2], aloc[0], grid_dimensions, lower_bound=lower_bound, upper_bound=upper_bound
    )
    return agent_grid_pos

def get_maps(scene_id, root_path):
    gmap_path = osp.join(root_path, f"{scene_id}_gmap.h5")
    with h5py.File(gmap_path, "r") as f:
        nav_map  = f['nav_map'][()]
        room_map = f['room_map'][()] > 0
        obj_maps = f['obj_maps'][()] > 0
        bounds = f['bounds'][()]

    recolor_map = np.array(
            [[255, 255, 255, 255], [128, 128, 128, 255], [0, 0, 0, 255]], dtype=np.uint8
    )
    nav_map = recolor_map[nav_map]
    grid_dimensions = (nav_map.shape[0], nav_map.shape[1])
    return nav_map, room_map, obj_maps, grid_dimensions, bounds

def get_raw_maps(scene_id, root_path):
    gmap_path = osp.join(root_path, f"{scene_id}_gmap.h5")
    with h5py.File(gmap_path, "r") as f:
        nav_map  = f['nav_map'][()]
        room_map = f['room_map'][()]
        obj_maps = f['obj_maps'][()]
        bounds = f['bounds'][()]

    grid_dimensions = (nav_map.shape[0], nav_map.shape[1])
    return nav_map, room_map, obj_maps, grid_dimensions, bounds

def load_panos(scene_name, pano_path):
    """
    Args:
        scene_name
        pano_path
    Return:
        panos [X * Y, K, K, 3]
    """
    pass

def to_grid(
    realworld_x: float,
    realworld_y: float,
    grid_resolution: Tuple[int, int],
    lower_bound, upper_bound
) -> Tuple[int, int]:
    """
    single point implementation
    """
    grid_size = (
        abs(upper_bound[2] - lower_bound[2]) / grid_resolution[0],
        abs(upper_bound[0] - lower_bound[0]) / grid_resolution[1],
    )
    grid_x = int((realworld_x - lower_bound[2]) / grid_size[0])
    grid_y = int((realworld_y - lower_bound[0]) / grid_size[1])
    return grid_x, grid_y

def from_grid(
    grid_x: int,
    grid_y: int,
    grid_resolution: Tuple[int, int],
    lower_bound, upper_bound
) -> Tuple[float, float]:
    """
    single point implementation
    """
    grid_size = (
        abs(upper_bound[2] - lower_bound[2]) / grid_resolution[0],
        abs(upper_bound[0] - lower_bound[0]) / grid_resolution[1],
    )
    realworld_x = lower_bound[2] + grid_x * grid_size[0]
    realworld_y = lower_bound[0] + grid_y * grid_size[1]
    return realworld_x, realworld_y


import timeit
def execution_time(method):
    """ decorator style """

    def time_measure(*args, **kwargs):
        ts = timeit.default_timer()
        result = method(*args, **kwargs)
        te = timeit.default_timer()

        print(f'Excution time of method {method.__qualname__} is {te - ts} seconds.')
        #print(f'Excution time of method {method.__name__} is {te - ts} seconds.')
        return result

    return time_measure
