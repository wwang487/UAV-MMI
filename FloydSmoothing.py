# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 21:16:58 2022

@author: bbean
"""
import numpy as np
import copy

def isBlockedPosition(blocked_map, pos):
    x, y = pos
    if x < 0 or y < 0 or x >= len(blocked_map[0]) or y >= len(blocked_map):
        return True  # position is out of bounds
    return blocked_map[y][x] == float('inf')  # position is blocked

def isBlockedSegment(x0, y0, x1, y1):
    dx = x1 - x0
    dy = y1 - y0
    steps = max(abs(dx), abs(dy))
    x_step = dx / steps if steps > 0 else 0
    y_step = dy / steps if steps > 0 else 0
    x = x0
    y = y0
    for i in range(int(steps) + 1):
        if isBlockedPosition((int(x), int(y))):
            return True
        x += x_step
        y += y_step
    return False

def Collinear(x, y, z):
    return (y[1] - x[1])*(z[0] - x[0]) == (y[0] - x[0])*(z[1] - x[1])

def FloydSmoothing(orig_path, bw_map):
    # orig_path is a 2-D array with dim_1 ~ n points, dim_2 ~ x and y
    
    if orig_path.shape[0] == 0:
        return orig_path
    else:
        path_len = orig_path.shape[0]
        if path_len <= 2:
            return orig_path
        else:            
            # First, remove extra path
            start_ind, delete_ind = 0, []
            for i in range(1, path_len - 1):
                x, y, z = orig_path[start_ind, :], orig_path[i, :], orig_path[i + 1, :]
                if Collinear(x, y, z):
                    delete_ind.append(i)                   
                else:
                    start_ind = i
            temp_path = copy.deepcopy(orig_path)
            delete_len = len(delete_ind)
            for j in range(delete_len - 1, -1, -1):
                temp_path = np.delete(temp_path, delete_ind[j], 0)    
            # print(temp_path)
            # Then, delete nonnecessary turns
            second_delete_ind = []
            path_len = temp_path.shape[0]
            if path_len <= 2:
                return temp_path
            i0, j0 = 0, 1
            while i0 < path_len - 1 and j0 < path_len - 1:
                for j0 in range(i0 + 2, path_len):
                    x1, y1, x2, y2 = temp_path[i0, 0], temp_path[i0, 1], temp_path[j0, 0],\
                                     temp_path[j0, 1]
                    covered_array = ExtractCoveredPixels(x1, y1, x2, y2)
                    is_block = CheckStraightLineBlocked(covered_array, bw_map)
                    # print([x1, y1, x2, y2, is_block])
                    if not is_block:
                        second_delete_ind.append(j0 - 1)
                    else:
                        i0 = j0 - 1
                        break
                    
            second_delete_len = len(second_delete_ind)
            # print(second_delete_ind)
            for k0 in range(second_delete_len - 1, -1, -1):
                # print(k0)
                temp_path = np.delete(temp_path, second_delete_ind[k0], 0)    
            return temp_path
            
def ExtractCoveredPixels(x1, y1, x2, y2):
    x_min, x_max, y_min, y_max = int(np.floor(min(x1, x2))), int(np.ceil(max(x1, x2))), \
                                 int(np.floor(min(y1, y2))), int(np.ceil(max(y1, y2)))
    
    if x_min == x_max:
        return np.array([[x_min]*(y_max - y_min + 1), list(range(y_min, y_max + 1))])
    elif y_min == y_max:
        return np.array([[y_min]*(x_max - x_min + 1), list(range(x_min, x_max + 1))])
    else:
        m = (y_max - y_min) / (x_max - x_min)
        c = y_min - m * x_min
        x_res, y_res = [], []
        prev_y = y_min
        for x in range(x_min + 1, x_max + 1):
            curr_y = int(np.ceil(m * x + c))
            if curr_y == prev_y:
                x_res, y_res = x_res + [x], y_res + [curr_y]
            else:
                x_res, y_res = x_res + [x] * (curr_y - prev_y + 1), y_res + list(range(prev_y, curr_y + 1))
                prev_y = curr_y
        return np.array([x_res, y_res])        

def CheckStraightLineBlocked(covered_array, bw_map):
    point_num = covered_array.shape[-1]
    res = False
    for i in range(point_num):
        if bw_map[covered_array[0, i], covered_array[1, i]] > 0 or bw_map[covered_array[0, i], covered_array[1, i]] == np.inf:
            res = True
            break
    return res
                