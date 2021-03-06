"""
extract instance around point
"""
from this import d
import numpy as np
import math
from collections import defaultdict
from PIL import Image
from constants import tab10_colors_rgba
from utils.map_tools import draw_point, execution_time, get_agent_orientation
import quaternion
from scipy.spatial import distance
import time

def scaler(bound, target_range, value):
    value = np.clip(value, bound[0], bound[1])
    v_std = (value-bound[0]) / (bound[1]-bound[0])
    return v_std * (target_range[1] - target_range[0]) + target_range[0]

@execution_time
def get_surrounding_objs(obj_map, nav_map, agent_grid_pos, radius, floor_idx=1, exemption_list=[1,16,38,39], meter_per_pixel=0.05):
    """
    Args:
        obj_map: raw obj map (denote pixel by instance id)
        radius: in meter
    """
    r, c, arot = agent_grid_pos # rot in radian
    rot = get_agent_orientation(arot)

    assert nav_map[r,c] > 0 or obj_map[r,c,floor_idx] > 0
    h, w = nav_map.shape[:2]
    bound = int(round(radius/meter_per_pixel))

    lr = max(0, r - bound) 
    lc = max(0, c - bound) 
    hr = min(h, r + bound)
    hc = min(w, c + bound)
    
    result_dict = defaultdict(list)
    relative_dict = defaultdict(list)
    for layer_idx in range(obj_map.shape[2]):
        if layer_idx in exemption_list:
            continue
        ins_set = np.unique(obj_map[lr:hr, lc:hc,layer_idx])
        ins_set = np.delete(ins_set, 0)
        if len(ins_set)==0:
            continue
            
        for i, ins in enumerate(ins_set):
            coords = np.where(obj_map[lr:hr, lc:hc, layer_idx] == ins)
            #coords = tuple(zip(*coords))
            cr, cc = np.median(coords[0])+lr, np.median(coords[1])+lc
            result_dict[layer_idx].append((cr, cc, ins-5))                               # grid row, grid col, instance id

            # rotate to egocentric coordinate (first coord is along agent head, second coord is perpendicular to agent head in clockwise direction, )
            rel_cr, rel_cc = rotate((0,0), (cr-r, cc-c), -rot)
            # standardize relative position to [-1,1]
            rel_cr = scaler((-bound, bound), (-1,1), rel_cr)
            rel_cc = scaler((-bound, bound), (-1,1), rel_cc)
            
            relative_dict[layer_idx].append((rel_cr, rel_cc, math.atan2(cc-c, cr-r) - rot, ins-5)) # relative row, relative col, angel, instance id
    return result_dict, relative_dict

@execution_time
def get_room_compass(room_map, nav_map, agent_grid_pos, radius=None, num_chunks=12, exemption_list=[], meter_per_pixel=0.05):
    """
    room_status: {
            0: (room_type, dist=0, (r,c, 0)), 1: (room_type, dist1, (r,c, ang1)), 
            2: (room_type, dist2, (r,c,ang2)) ... 12: (room_type, dist12, (r,c,ang12))
        }
    
    """
    r, c, arot = agent_grid_pos # rot in radian
    rot = get_agent_orientation(arot)
    h, w = nav_map.shape[:2]

    if radius is not None:
        bound = int(round(radius/meter_per_pixel))

        lr = max(0, r - bound) 
        lc = max(0, c - bound) 
        hr = min(h, r + bound)
        hc = min(w, c + bound)
    else:
        lr=0;lc=0;hr=h,hc=w

    # creating room status chunks
    chunk_radian = (np.pi * 2) / num_chunks
    current_room_idx = np.argmax(room_map[r,c,:])
    instance_to_skip = room_map[r,c,current_room_idx]
    room_status = { 0: (current_room_idx, 0, (0, 0, 0))}

    for layer_idx in range(room_map.shape[2]):
        if layer_idx in exemption_list:
            continue
        ins_set = np.unique(room_map[lr:hr, lc:hc, layer_idx])
        ins_set = np.delete(ins_set, 0)
        if len(ins_set)==0:
            continue

        for i, ins in enumerate(ins_set):
            if ins == instance_to_skip:
                continue
            coords = np.where(room_map[lr:hr,lc:hc,layer_idx] == ins)
            cr, cc = np.median(coords[0]) + lr, np.median(coords[1]) + lc

            rel_ang = math.atan2(cc-c, cr-r) - rot
            if rel_ang > np.pi:
                rel_ang -= 2*np.pi
            elif rel_ang < -np.pi:
                rel_ang += 2*np.pi

            rel_chunk = int((rel_ang + np.pi ) / chunk_radian)
            rel_dist = distance.euclidean((cr, cc), (r,c)) * meter_per_pixel

            curr_status = room_status.get(rel_chunk+1, (-1, 200, ()))
           
            # distance comparation
            if rel_dist < curr_status[1]:
                rel_cr, rel_cc = rotate((0,0), (cr-r, cc-c), -rot)
                room_status[rel_chunk+1] = (layer_idx, rel_dist, (rel_cr, rel_cc, rel_ang))
    return room_status
            

def rotate(origin, point, angle):
    """
    Rotate a point counter-clockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy

def draw_instances(instance_dict, agent_grid_pos, size):
    r, c, arot = agent_grid_pos 
    rot = get_agent_orientation(arot)
    overlay = Image.new('RGBA', size, (255,0,0)+(0,))
    for layer_idx, coord_list in instance_dict.items():
        for i, ins_coord in enumerate(coord_list):
            cr, cc = ins_coord[0], ins_coord[1]
            draw_point(overlay, cc, cr, 5, color=tuple(tab10_colors_rgba[layer_idx%10]), text=str(layer_idx))
            
            # verify relative rotation
            # cr, cc = ins_coord[0]*60 + r, ins_coord[1]*60 + c 
            # cr_2, cc_2 = rotate((r,c), (cr,cc), rot)
            # draw_point(overlay, cc_2, cr_2, 10, color=tuple(tab10_colors_rgba[layer_idx%10]), text=str(layer_idx))
            # return overlay
    return overlay

def draw_room_instances(instance_dict, agent_grid_pos, size):
    r, c, arot = agent_grid_pos 
    rot = get_agent_orientation(arot)
    overlay = Image.new('RGBA', size, (255,0,0)+(0,))
    for chunk_idx, info in instance_dict.items():
        # verify relative rotation
        print(chunk_idx, info)
        cr, cc = info[-1][0] + r, info[-1][1] + c 
        cr_2, cc_2 = rotate((r,c), (cr,cc), rot)
        draw_point(overlay, cc_2, cr_2, 10, color=tuple(tab10_colors_rgba[chunk_idx%10]), text=str(info[0]))
    return overlay