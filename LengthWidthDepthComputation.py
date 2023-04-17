# -*- coding: utf-8 -*-
"""
Created on Sat Mar 12 08:23:09 2022

@author: bbean
"""
#%% This cell deals with package import.
import cv2
from copy import deepcopy
import numpy as np
import pandas as pd
import math
from osgeo import gdal
from osgeo import osr
import re
from urllib import request
from urllib import error
from bs4 import BeautifulSoup
from dateutil.relativedelta import relativedelta
import datetime
import time
from osgeo.gdalconst import GA_ReadOnly
import os
from shapely.geometry import Polygon
import rasterio
from rasterio import mask
import seaborn as sns

from pylab import *

import statsmodels.api as sm
import matplotlib.pyplot as plt

os.environ['GDAL_DATA'] = 'D:\\software\\Anaconda3\\envs\\GeoAnalysis\\Library\\share\\gdal'
os.environ['PROJ_LIB'] = 'D:\\software\\Anaconda3\\envs\\GeoAnalysis\\Library\\share\\proj'

def CheckIfFolderExist(filepath):
    if os.path.exists(f'{filepath}'):
        pass
    else:
        os.mkdir(f'{filepath}') 

#%% This cell computes width-related information.
def DistanceBetweenTwoPoints(x1, y1, x2, y2):
    dist = math.sqrt(math.pow((x2 - x1), 2) + math.pow((y2 - y1), 2))
    return dist

def UpdateListWithMinVal(list0, m):
    res = []
    for l in list0:
        res.append(l - m)
    return res
    
def UpdateListWithAddingMinVal(list0, m):
    res = []
    for l in list0:
        res.append(l + m)
    return res

def DistanceBetweenPointAndSegment(px, py, x1, y1, x2, y2):
    line_length = DistanceBetweenTwoPoints(x1, y1, x2, y2)
    if line_length < 0.00000001:
        return 9999
    else:
        u1 = (((px - x1) * (x2 - x1)) + ((py - y1) * (y2 - y1)))
        u = u1 / (line_length * line_length)
        if (u < 0.00001) or (u > 1):
            i1 = DistanceBetweenTwoPoints(px, py, x1, y1)
            i2 = DistanceBetweenTwoPoints(px, py, x2, y2)
            if i1 > i2:
                distance = i2
                ix, iy = x2, y2
            else:
                distance = i1
                ix, iy = x1, y1
        else:
            ix = x1 + u * (x2 - x1)
            iy = y1 + u * (y2 - y1)
            distance = DistanceBetweenTwoPoints(px, py, ix, iy)
        return distance, ix, iy
    
def DistanceBetweenPointAndSegmentWithEyot(px, py, x1, y1, x2, y2, eyot_left_x_list, eyot_left_y_list, \
                                           eyot_right_x_list, eyot_right_y_list):
    line_length = DistanceBetweenTwoPoints(x1, y1, x2, y2)
    left_eyot_point_num = len(eyot_left_x_list)
    right_eyot_point_num = len(eyot_right_x_list)
    if line_length < 0.00000001:
        return 9999
    else:
        u1 = (((px - x1) * (x2 - x1)) + ((py - y1) * (y2 - y1)))
        u = u1 / (line_length * line_length)
        
        if (u < 0.00001) or (u > 1):
            i1 = DistanceBetweenTwoPoints(px, py, x1, y1)
            i2 = DistanceBetweenTwoPoints(px, py, x2, y2)
            if i1 > i2:
                distance = i2
                ix, iy = x2, y2
            else:
                distance = i1
                ix, iy = x1, y1
        else:
            ix = x1 + u * (x2 - x1)
            iy = y1 + u * (y2 - y1)
            distance = DistanceBetweenTwoPoints(px, py, ix, iy)
        
        p1, p2 = [px, py], [ix, iy]
        
        for j in range(left_eyot_point_num - 1):
            p3 = [eyot_left_x_list[j], eyot_left_y_list[j]]
            p4 = [eyot_left_x_list[j + 1], eyot_left_y_list[j + 1]]
            if IsTwoSegmentIntersected(p1, p2, p3, p4):
                left_intersect = computeIntersectPoint(p1, p2, p3, p4)                
                for k in range(right_eyot_point_num - 1):
                    p3 = [eyot_right_x_list[k], eyot_right_y_list[k]]
                    p4 = [eyot_right_x_list[k + 1], eyot_right_y_list[k + 1]]
                    if IsTwoSegmentIntersected(p1, p2, p3, p4):
                        right_intersect = computeIntersectPoint(p1, p2, p3, p4)
                        distance = distance - DistanceBetweenTwoPoints(left_intersect[0], \
                                   left_intersect[1], right_intersect[0], right_intersect[1]) 
                        break
                break
        return distance, ix, iy

def InterpolatePolyline(x_list, y_list, times):
    point_num = len(x_list)
    res_x = []
    res_y = []
    for i in range(point_num - 1):
        s_x = x_list[i]
        s_y = y_list[i]
        t_x = x_list[i + 1]
        t_y = y_list[i + 1]
        res_x.append(s_x)
        res_y.append(s_y)
        d_x = (t_y - s_y) / times
        d_y = (t_x - s_x) / times
        for j in range(times - 1):
            res_x.append(s_x + (j + 1) * d_x)
            res_y.append(s_y + (j + 1) * d_y)
    # res_x.append(x_list[i + 1])
    # res_y.append(y_list[i + 1])
    return res_x, res_y       

def DistanceBetweenAPointAndOtherBank(p_x, p_y, x_list, y_list):
    bank_num = len(x_list)
    curr_dist = 999999
    res_x = -999999
    res_y = -999999
    for i in range(bank_num - 1):
        x_0 = x_list[i]
        y_0 = y_list[i]
        x_1 = x_list[i + 1]
        y_1 = y_list[i + 1]
        
        d, i_x, i_y = DistanceBetweenPointAndSegment(p_x, p_y, x_0, y_0, x_1, y_1)
        
        if d < curr_dist:
            curr_dist = d
            res_x = i_x
            res_y = i_y
        elif d > 15 * curr_dist:
            break
    #print([p_x, p_y, x_0, y_0, x_1, y_1, d, res_x, res_y])
    return curr_dist, res_x, res_y

def DistanceBetweenAPointAndOtherBankWithEyot(p_x, p_y, x_list, y_list, left_eyot_x, left_eyot_y, right_eyot_x, right_eyot_y):
    bank_num = len(x_list)
    curr_dist = 999999
    res_x = -999999
    res_y = -999999
    p30 = [left_eyot_x[0], left_eyot_y[0]]
    p40 = [left_eyot_x[-1], left_eyot_y[-1]]
    for i in range(bank_num - 1):
        x_0 = x_list[i]
        y_0 = y_list[i]
        x_1 = x_list[i + 1]
        y_1 = y_list[i + 1]
        if IsTwoSegmentIntersected([p_x, p_y], [x_0, y_0], p30, p40) or IsTwoSegmentIntersected([p_x, p_y], [x_1, y_1], p30, p40):
            d, i_x, i_y = DistanceBetweenPointAndSegmentWithEyot(p_x, p_y, x_0, y_0, x_1, y_1, \
                                                                 left_eyot_x, left_eyot_y, right_eyot_x, right_eyot_y)
        else:
            d, i_x, i_y = DistanceBetweenPointAndSegment(p_x, p_y, x_0, y_0, x_1, y_1)
        
        if d < curr_dist:
            curr_dist = d
            res_x = i_x
            res_y = i_y
        elif d > 15 * curr_dist:
            break
    #print([p_x, p_y, x_0, y_0, x_1, y_1, d, res_x, res_y])
    return curr_dist, res_x, res_y
        
def DistanceBetweenLeftAndRightBank(left_x, left_y, right_x, right_y, min_num):
    # We assume that left bank is selected as the reference bank
    print('start!')
    left_num = len(left_x)
    right_num = len(right_x)
    left_times = 3 * int(np.ceil(min_num / left_num)) if left_num < min_num else 1
    right_times = 3 * int(np.ceil(min_num / right_num)) if right_num < min_num else 1
    print([left_num, min_num, left_times])
    print([right_num, min_num, right_times])
    if left_times > 1:    
        new_left_x, new_left_y = InterpolatePolyline(left_x, left_y, left_times)
    else:
        new_left_x, new_left_y = left_x, left_y
    if right_times > 1:
        new_right_x, new_right_y = InterpolatePolyline(right_x, right_y, right_times)
    else:
        new_right_x, new_right_y = right_x, right_y 
    width_res = []
    mid_x_res = []
    mid_y_res = []
    right_x_res = []
    right_y_res = []
    new_left_num = len(new_left_x)
    for i in range(new_left_num):
        px, py = new_left_x[i], new_left_y[i]
        d, res_x, res_y = DistanceBetweenAPointAndOtherBank(px, py, \
                                        new_right_x, new_right_y)
        width_res.append(d)
        mid_x = (px + res_x) / 2
        mid_y = (py + res_y) / 2
        mid_x_res.append(mid_x)
        mid_y_res.append(mid_y)
        right_x_res.append(res_x)
        right_y_res.append(res_y)
    return width_res, mid_x_res, mid_y_res, new_left_x, new_left_y, \
            right_x_res, right_y_res

def DistanceBetweenLeftAndRightBankWithEyot(left_x, left_y, right_x, right_y, \
                                            left_eyot_x, left_eyot_y, right_eyot_x, right_eyot_y, min_num):
    # We assume that left bank is selected as the reference bank
    print('start!')
    left_num = len(left_x)
    right_num = len(right_x)
    left_times = 3 * int(np.ceil(min_num / left_num)) if left_num < min_num else 1
    right_times = 3 * int(np.ceil(min_num / right_num)) if right_num < min_num else 1
    print([left_num, min_num, left_times])
    print([right_num, min_num, right_times])
    if left_times > 1:    
        new_left_x, new_left_y = InterpolatePolyline(left_x, left_y, left_times)
    else:
        new_left_x, new_left_y = left_x, left_y
    if right_times > 1:
        new_right_x, new_right_y = InterpolatePolyline(right_x, right_y, right_times)
    else:
        new_right_x, new_right_y = right_x, right_y 
    width_res = []
    mid_x_res = []
    mid_y_res = []
    right_x_res = []
    right_y_res = []
    new_left_num = len(new_left_x)
    
    for i in range(new_left_num):
        px, py = new_left_x[i], new_left_y[i]
        d, res_x, res_y = DistanceBetweenAPointAndOtherBankWithEyot(px, py, new_right_x, new_right_y,\
                                                 left_eyot_x, left_eyot_y, right_eyot_x, right_eyot_y)
        width_res.append(d)
        mid_x = (px + res_x) / 2
        mid_y = (py + res_y) / 2
        mid_x_res.append(mid_x)
        mid_y_res.append(mid_y)
        right_x_res.append(res_x)
        right_y_res.append(res_y)
    return width_res, mid_x_res, mid_y_res, new_left_x, new_left_y, \
            right_x_res, right_y_res

def updateWidthByEyot(width_res, left_x_list, left_y_list, right_x_list, right_y_list,
        eyot_left_x_list, eyot_left_y_list, eyot_right_x_list, eyot_right_y_list):
    bank_point_num = len(left_x_list)
    left_eyot_point_num = len(eyot_left_x_list)
    right_eyot_point_num = len(eyot_right_x_list)
    intersect_ind, left_xs, left_ys, right_xs, \
            right_ys = [],[],[],[],[]
    dist = deepcopy(width_res)
    for i in range(bank_point_num):
        p1 = [left_x_list[i], left_y_list[i]]
        p2 = [right_x_list[i], right_y_list[i]]
        p30 = [eyot_left_x_list[0], eyot_left_y_list[0]]
        p40 = [eyot_left_x_list[-1], eyot_left_y_list[-1]]
        if IsTwoSegmentIntersected(p1, p2, p30, p40):
            print(True)
        if i == 0:
            prev_intersected = False
            curr_intersected = False
        # else:
        #    if prev_intersected and not curr_intersected:
        #        break
        #    elif not prev_intersected and curr_intersected:
        #        prev_intersected = curr_intersected
        
        curr_intersected = False 
        
        for j in range(left_eyot_point_num - 1):
            p3 = [eyot_left_x_list[j], eyot_left_y_list[j]]
            p4 = [eyot_left_x_list[j + 1], eyot_left_y_list[j + 1]]
            if IsTwoSegmentIntersected(p1, p2, p3, p4):
                left_intersect = computeIntersectPoint(p1, p2, p3, p4)                
                for k in range(right_eyot_point_num - 1):
                    p3 = [eyot_right_x_list[k], eyot_right_y_list[k]]
                    p4 = [eyot_right_x_list[k + 1], eyot_right_y_list[k + 1]]
                    if IsTwoSegmentIntersected(p1, p2, p3, p4):
                        right_intersect = computeIntersectPoint(p1, p2, p3, p4)
                        curr_intersected = True
                        break
                break
                
        if curr_intersected:
            print([i, j, k])
            print(p1)
            print(p2)
            print(p3)
            print(p4)
            
            intersect_ind.append(i)
            left_xs.append(left_intersect[0])
            left_ys.append(left_intersect[1])
            right_xs.append(right_intersect[0])
            right_ys.append(right_intersect[1])
            dist[i] = dist[i] - DistanceBetweenTwoPoints(left_intersect[0], \
                    left_intersect[1], right_intersect[0], right_intersect[1]) 
            
        prev_intersected = curr_intersected
    return intersect_ind, left_xs, left_ys, right_xs, right_ys, dist

def ExtractFixNumberOfBankPoints(x_list, y_list, point_num):
    cul_len = 0
    x_num = len(x_list)
    for i in range(x_num - 1):
        x1, x2, y1, y2 = x_list[i], x_list[i + 1], y_list[i], y_list[i + 1]
        cul_len = cul_len + DistanceBetweenTwoPoints(x1, y1, x2, y2)
    len_interval = cul_len / point_num
    print([cul_len, len_interval])
    curr_d = len_interval
    j = 0
    x_res, y_res = [x_list[j]], [y_list[j]]
    
    curr_x, curr_y = x_list[j], y_list[j]
    next_x, next_y = x_list[j + 1], y_list[j + 1]
    while len(x_res) < point_num and j < x_num:
        total_d = DistanceBetweenTwoPoints(curr_x, curr_y, next_x, next_y)
        if curr_d == total_d:
            x_res.append(next_x)
            y_res.append(next_y)
            curr_d = len_interval
            j = j + 1
            if len(x_res) < point_num and j < x_num:
                curr_x, curr_y, next_x, next_y = x_list[j], y_list[j], x_list[j + 1], y_list[j + 1]
        elif curr_d < total_d:
            total_dx, total_dy = next_x - curr_x, next_y - curr_y
            ratio = curr_d / total_d
            x_res.append(curr_x + total_dx * ratio)
            y_res.append(curr_y + total_dy * ratio)
            curr_x, curr_y = x_res[-1], y_res[-1]
            curr_d = len_interval
        else:
            curr_d = curr_d - total_d
            j = j + 1
            if len(x_res) < point_num and j < x_num:
                curr_x, curr_y, next_x, next_y = x_list[j], y_list[j], x_list[j + 1], y_list[j + 1]
    if x_res[-1] != x_list[-1] or y_res[-1] != y_list[-1]:
         x_res.append(x_list[-1])
         y_res.append(y_list[-1])
            
    # for j in range(x_num - 1): 
    #     x1, x2, y1, y2 = x_list[j], x_list[j + 1], y_list[j], y_list[j + 1]
    #     total_d = DistanceBetweenTwoPoints(x1, y1, x2, y2)
    #     # print(total_d)
    #     temp_d = curr_d - total_d
        
    #     if temp_d > 0:
    #         curr_d = temp_d
    #     elif temp_d == 0:
    #         curr_d = len_interval
    #         x_res.append(x_list[j + 1])
    #         y_res.append(y_list[j + 1])
    #     else:
    #         total_dx, total_dy = x_list[j + 1] - x_list[j], y_list[j + 1] - y_list[j]
    #         start_x, start_y = x_list[j], y_list[j]
    #         while temp_d < 0:            
                               
    #             ratio = curr_d / total_d
    #             x_res.append(start_x + total_dx * ratio)
    #             y_res.append(start_y + total_dy * ratio)
    #             start_x, start_y = x_res[-1], y_res[-1]
    #             curr_d = len_interval
    #             temp_d = curr_d - DistanceBetweenTwoPoints(start_x, start_y, x2, y2)
    # if x_res[-1] != x_list[-1] or y_res[-1] != y_list[-1]:
    #     x_res.append(x_list[-1])
    #     y_res.append(y_list[-1])
    
    return x_res, y_res
        
def DistanceBetweenFixedLeftAndRightBankPointsWithEyot(left_x, left_y, right_x, right_y,\
               left_eyot_x, left_eyot_y, right_eyot_x, right_eyot_y):
    mid_x, mid_y, dist = [], [], []
    left_inter_x, right_inter_x, left_inter_y, right_inter_y = [], [], [], []
    point_num = len(left_x)
    if left_eyot_x and left_eyot_y and right_eyot_x and right_eyot_y:
        left_eyot_point_num = len(left_eyot_x)
        right_eyot_point_num = len(right_eyot_x)
        p30 = [left_eyot_x[0], left_eyot_y[0]]
        p40 = [left_eyot_x[-1], left_eyot_y[-1]]
        
        for i in range(point_num):
            mid_x.append((left_x[i] + right_x[i]) / 2)
            mid_y.append((left_y[i] + right_y[i]) / 2)
            temp_dist = DistanceBetweenTwoPoints(left_x[i], left_y[i], right_x[i], right_y[i])
            p1, p2 = [left_x[i], left_y[i]], [right_x[i], right_y[i]]
            if not IsTwoSegmentIntersected(p1, p2, p30, p40):
                dist.append(temp_dist)
                left_inter_x.append(-999999)
                right_inter_x.append(-999999)
                left_inter_y.append(-999999)
                right_inter_y.append(-999999)
            else:
                for j in range(left_eyot_point_num - 1):
                    p3 = [left_eyot_x[j], left_eyot_y[j]]
                    p4 = [left_eyot_x[j + 1], left_eyot_y[j + 1]]
                    if IsTwoSegmentIntersected(p1, p2, p3, p4):
                        left_intersect = computeIntersectPoint(p1, p2, p3, p4)                
                        for k in range(right_eyot_point_num - 1):
                            p3 = [right_eyot_x[k], right_eyot_y[k]]
                            p4 = [right_eyot_x[k + 1], right_eyot_y[k + 1]]
                            if IsTwoSegmentIntersected(p1, p2, p3, p4):
                                # print('aaaaa')
                                right_intersect = computeIntersectPoint(p1, p2, p3, p4)
                                temp_dist = temp_dist - DistanceBetweenTwoPoints(left_intersect[0], \
                                       left_intersect[1], right_intersect[0], right_intersect[1]) 
                                left_inter_x.append(left_intersect[0])
                                right_inter_x.append(right_intersect[0])
                                left_inter_y.append(left_intersect[1])
                                right_inter_y.append(right_intersect[1])
                                break
                        break
                dist.append(temp_dist)
    else:
        for i in range(point_num):
            mid_x.append((left_x[i] + right_x[i]) / 2)
            mid_y.append((left_y[i] + right_y[i]) / 2)
            dist.append(DistanceBetweenTwoPoints(left_x[i], left_y[i], right_x[i], right_y[i]))
    return mid_x, mid_y, dist, left_inter_x, right_inter_x, left_inter_y, right_inter_y

# I need to fix this with minimum, also, create another one compute cumulative
def getDistanceBetweenTransectsAndPolylines(left_x_list, left_y_list, right_x_list, right_y_list, feature_x_list, feature_y_list, choice = 0, dist_thresh = 50):
    # choice 0 means only find the first matching, choice 1 means find the minimum matching.
    res = []
    left_eyot_point_num = len(feature_x_list)
    trans_num = len(left_x_list)
    for i in range(trans_num):
        left_x, left_y, right_x, right_y = left_x_list[i], left_y_list[i], right_x_list[i], right_y_list[i]
        p1, p2 = [left_x, left_y], [right_x, right_y]
        A, B, C = getABCofTwoPointSegment(left_x, left_y, right_x, right_y)
        found = False
        temp_res = []
        for j in range(left_eyot_point_num - 1):
            p3 = [feature_x_list[j], feature_y_list[j]]
            p4 = [feature_x_list[j + 1], feature_y_list[j + 1]]
            D1 = A * p3[0] + B * p3[1] + C
            D2 = A * p4[0] + B * p4[1] + C
            if D1 * D2 <= 0:
                A1, B1, C1 = getABCofTwoPointSegment(p3[0], p3[1], p4[0], p4[1])
                ix, iy = getIntersectsOfTwoABCs(A, B, C, A1, B1, C1)           
                dist_1 = DistanceBetweenTwoPoints(ix, iy, p1[0], p1[1])
                dist_2 = DistanceBetweenTwoPoints(ix, iy, p2[0], p2[1])
                temp_res.append(min(dist_1, dist_2))
                found = True
                if choice == 0:
                    break    
                elif DistanceBetweenTwoPoints(p1[0], p1[1], p3[0], p3[1]) > dist_thresh and DistanceBetweenTwoPoints(p2[0], p2[1], p3[0], p3[1]) > dist_thresh:
                    break
        # if found:
        #     print('Found')
        if not found:
            res.append(0)
        else:
            res.append(min(temp_res))
    return res       

def getCumDistanceBetweenTransectsAndPolylines(left_x_list, left_y_list, right_x_list, right_y_list, feature_x_list, feature_y_list, dist_thresh = 50):
    res = []
    left_eyot_point_num = len(feature_x_list)
    trans_num = len(left_x_list)
    for i in range(trans_num):
        left_x, left_y, right_x, right_y = left_x_list[i], left_y_list[i], right_x_list[i], right_y_list[i]
        p1, p2 = [left_x, left_y], [right_x, right_y]
        A, B, C = getABCofTwoPointSegment(left_x, left_y, right_x, right_y)
        temp_res = []
        found = False
        for j in range(left_eyot_point_num - 1):
            p3 = [feature_x_list[j], feature_y_list[j]]
            p4 = [feature_x_list[j + 1], feature_y_list[j + 1]]
            D1 = A * p3[0] + B * p3[1] + C
            D2 = A * p4[0] + B * p4[1] + C
            if D1 * D2 <= 0:
                A1, B1, C1 = getABCofTwoPointSegment(p3[0], p3[1], p4[0], p4[1])
                ix, iy = getIntersectsOfTwoABCs(A, B, C, A1, B1, C1)           
                dist_1 = DistanceBetweenTwoPoints(ix, iy, p1[0], p1[1])
                dist_2 = DistanceBetweenTwoPoints(ix, iy, p2[0], p2[1])
                temp_res.append(min(dist_1, dist_2))
                found = True
                if DistanceBetweenTwoPoints(p1[0], p1[1], p3[0], p3[1]) > dist_thresh and DistanceBetweenTwoPoints(p2[0], p2[1], p3[0], p3[1]) > dist_thresh:
                    break
        if not found:
            res.append(0)
        else:
            if len(temp_res) == 0 or len(temp_res) == 1:
                res.append(temp_res[0])
            else:
                temp_res.sort()
                cum_dist = 0
                for m in range(0, len(temp_res), 2):
                    if m == 0:
                        cum_dist = cum_dist + temp_res[0]
                    else:
                        cum_dist = cum_dist + temp_res[m] - temp_res[m - 1]
                res.append(cum_dist)            
    return res 

def RefineFixedWidthResBasedOnRefBank(fix_left_x, fix_left_y, fix_right_x, fix_right_y, stream_width, \
                        left_island_x, right_island_x, left_island_y, right_island_y, left_length, right_length, \
                        start_ind, end_ind, step = 30, inter_num = 5):
    if left_length <= right_length:
        ref_x, ref_y, check_x, check_y = fix_left_x, fix_left_y, fix_right_x, fix_right_y
    else:
        ref_x, ref_y, check_x, check_y = fix_right_x, fix_right_y, fix_left_x, fix_left_y
    
    new_width, new_check_x, new_check_y = [], [], []
    new_width.append(stream_width[0])
    new_check_x.append(check_x[0])
    new_check_y.append(check_y[0])
    
    for i in range(1, len(ref_x) - 1):
        if i < start_ind or i > end_ind:
            new_width.append(stream_width[i])
            new_check_x.append(check_x[i])
            new_check_y.append(check_y[i])
            continue
        elif left_island_x[i] >= 0 and right_island_x[i] >= 0 and left_island_y[i] >= 0 and right_island_y[i] >= 0:
            new_width.append(stream_width[i])
            new_check_x.append(check_x[i])
            new_check_y.append(check_y[i])
            continue
        else:
            check_widths = []
            curr_ref_x, curr_ref_y = ref_x[i], ref_y[i]
            start_check_ind, end_check_ind = max(i - step, 0), min(i + step, len(ref_x) - 1)
            # print([start_check_ind, end_check_ind])
            for j in range(start_check_ind, end_check_ind + 1):
                curr_check_x, curr_check_y = check_x[j], check_y[j]
                check_widths.append(DistanceBetweenTwoPoints(curr_ref_x, curr_ref_y, curr_check_x, curr_check_y))
            min_width_ind = np.argmin(check_widths)
            second_check_widths, second_check_x, second_check_y = [], [], []
            second_start_check_ind, second_end_check_ind = max(min_width_ind - 1, 0), min(min_width_ind + 1, len(check_widths) - 1)
            for k in range(second_start_check_ind, second_end_check_ind):
                
                second_start_x, second_start_y = check_x[start_check_ind + k], check_y[start_check_ind + k]
                second_end_x, second_end_y = check_x[start_check_ind + k + 1], check_y[start_check_ind + k + 1]
                
                for n in range(inter_num):
                    weight_1, weight_2 = (1 - n / inter_num), n / inter_num
                    curr_second_check_x, curr_second_check_y = second_start_x * weight_1 + second_end_x * weight_2,\
                                                               second_start_y * weight_1 + second_end_y * weight_2
                    second_check_widths.append(DistanceBetweenTwoPoints(curr_ref_x, curr_ref_y, curr_second_check_x, curr_second_check_y))
                    second_check_x.append(curr_second_check_x)
                    second_check_y.append(curr_second_check_y)
                if k == second_end_check_ind:
                    curr_second_check_x, curr_second_check_y = second_end_x, second_end_y
                    second_check_widths.append(DistanceBetweenTwoPoints(curr_ref_x, curr_ref_y, curr_second_check_x, curr_second_check_y))
                    second_check_x.append(curr_second_check_x)
                    second_check_y.append(curr_second_check_y)
            
            min_second_width = min(second_check_widths)
            # print(min_second_width)
            min_second_width_ind = np.argmin(second_check_widths)
            new_width.append(min_second_width)
            new_check_x.append(second_check_x[min_second_width_ind])
            new_check_y.append(second_check_y[min_second_width_ind])
    new_width.append(stream_width[-1])
    new_check_x.append(check_x[-1])
    new_check_y.append(check_y[-1])
    new_mid_x, new_mid_y = [], []
    for c in range(len(new_check_x)):
        new_mid_x.append((new_check_x[c] + ref_x[c]) / 2)
        new_mid_y.append((new_check_y[c] + ref_y[c]) / 2)
    if left_length <= right_length:
        new_left_x, new_right_x, new_left_y, new_right_y = ref_x, new_check_x, ref_y, new_check_y
    else:
        new_right_x, new_left_x, new_right_y, new_left_y = ref_x, new_check_x, ref_y, new_check_y
    return new_mid_x, new_mid_y, new_width, new_left_x, new_left_y, new_right_x, new_right_y

def RefineFixedWidthResBasedOnRotateRays(fix_left_x, fix_left_y, fix_right_x, fix_right_y, stream_width, \
                        left_island_x, right_island_x, left_island_y, right_island_y, mid_x, mid_y, \
                        start_ind, end_ind, step = 30, inter_num = 5):   
    
    new_width, new_left_x, new_left_y, new_right_x, new_right_y = [], [], [], [], []
    new_width.append(stream_width[0])
    new_left_x.append(fix_left_x[0])
    new_left_y.append(fix_left_y[0])
    new_right_x.append(fix_right_x[0])
    new_right_y.append(fix_right_y[0])
    
    for i in range(1, len(mid_x) - 1):
        if i < start_ind or i > end_ind:
            new_width.append(stream_width[i])
            new_left_x.append(fix_left_x[i])
            new_left_y.append(fix_left_y[i])
            new_right_x.append(fix_right_x[i])
            new_right_y.append(fix_right_y[i])
            continue
        elif left_island_x[i] >= 0 and right_island_x[i] >= 0 and left_island_y[i] >= 0 and right_island_y[i] >= 0:
            new_width.append(stream_width[i])
            new_left_x.append(fix_left_x[i])
            new_left_y.append(fix_left_y[i])
            new_right_x.append(fix_right_x[i])
            new_right_y.append(fix_right_y[i])
            continue
        else:
            start_check_ind, end_check_ind = max(i - step, 0), min(i + step, len(mid_x) - 1)
            check_mid_x, check_mid_y = mid_x[i], mid_y[i]
            check_widths = []
            for j in range(start_check_ind, end_check_ind + 1):
                check_left_x, check_left_y = fix_left_x[j], fix_left_y[j]
                A, B, C = getABCofTwoPointSegment(check_left_x, check_left_y, check_mid_x, check_mid_y)
                found = False
                for k in range(start_check_ind, end_check_ind):
                    p3 = [fix_right_x[k], fix_right_y[k]]
                    p4 = [fix_right_x[k + 1], fix_right_y[k + 1]]
                    D1 = A * p3[0] + B * p3[1] + C
                    D2 = A * p4[0] + B * p4[1] + C
                    if D1 * D2 <= 0:
                        A1, B1, C1 = getABCofTwoPointSegment(p3[0], p3[1], p4[0], p4[1])
                        
                        ix, iy = getIntersectsOfTwoABCs(A, B, C, A1, B1, C1)  
                        
                        dist_1 = DistanceBetweenTwoPoints(ix, iy, check_left_x, check_left_y)
                        check_widths.append(dist_1)
                        found = True
                        break
                if not found:
                    check_widths.append(999999)
            if not check_widths:
                new_width.append(stream_width[i])
                new_left_x.append(fix_left_x[i])
                new_left_y.append(fix_left_y[i])
                new_right_x.append(fix_right_x[i])
                new_right_y.append(fix_right_y[i])
                continue
            
            min_width_ind = np.argmin(check_widths)

            second_check_widths, second_check_left_x, second_check_left_y, second_check_right_x, second_check_right_y\
                                                   = [], [], [], [], []
            second_start_check_ind, second_end_check_ind = max(min_width_ind - 1, 0), min(min_width_ind + 1, len(check_widths) - 1)
            for sk in range(second_start_check_ind, second_end_check_ind):
                
                second_start_x, second_start_y = fix_left_x[start_check_ind + sk], fix_left_y[start_check_ind + sk]
                second_end_x, second_end_y = fix_left_x[start_check_ind + sk + 1], fix_left_y[start_check_ind + sk + 1]

                for n in range(inter_num):
                    weight_1, weight_2 = (1 - n / inter_num), n / inter_num
                    curr_second_check_x, curr_second_check_y = second_start_x * weight_1 + second_end_x * weight_2,\
                                                               second_start_y * weight_1 + second_end_y * weight_2
                    
                    A, B, C = getABCofTwoPointSegment(curr_second_check_x, curr_second_check_y, check_mid_x, check_mid_y)
                    found = False
                    for sj in range(start_check_ind, end_check_ind):
                        
                        p3 = [fix_right_x[sj], fix_right_y[sj]]
                        p4 = [fix_right_x[sj + 1], fix_right_y[sj + 1]]
                        D1 = A * p3[0] + B * p3[1] + C
                        D2 = A * p4[0] + B * p4[1] + C
                
                        if D1 * D2 <= 0:
                            A1, B1, C1 = getABCofTwoPointSegment(p3[0], p3[1], p4[0], p4[1])
                            ix, iy = getIntersectsOfTwoABCs(A, B, C, A1, B1, C1)           
                            dist_1 = DistanceBetweenTwoPoints(ix, iy, curr_second_check_x, curr_second_check_y)
                            second_check_widths.append(dist_1)
                            second_check_left_x.append(curr_second_check_x)
                            second_check_left_y.append(curr_second_check_y)
                            second_check_right_x.append(ix)
                            second_check_right_y.append(iy)
                            found = True
                            break
                    if not found:
                        continue

                if sk == second_end_check_ind:
                    curr_second_check_x, curr_second_check_y = second_end_x, second_end_y
                    A, B, C = getABCofTwoPointSegment(curr_second_check_x, curr_second_check_y, check_mid_x, check_mid_y)
                    found = False
                    for sn in range(start_check_ind, end_check_ind):
                        p3 = [fix_right_x[sn], fix_right_y[sn]]
                        p4 = [fix_right_x[sn + 1], fix_right_y[sn + 1]]
                        D1 = A * p3[0] + B * p3[1] + C
                        D2 = A * p4[0] + B * p4[1] + C
                        if D1 * D2 <= 0:
                            A1, B1, C1 = getABCofTwoPointSegment(p3[0], p3[1], p4[0], p4[1])
                            ix, iy = getIntersectsOfTwoABCs(A, B, C, A1, B1, C1)           
                            dist_1 = DistanceBetweenTwoPoints(ix, iy, curr_second_check_x, curr_second_check_y)
                            second_check_widths.append(dist_1)
                            second_check_left_x.append(curr_second_check_x)
                            second_check_left_y.append(curr_second_check_y)
                            second_check_right_x.append(ix)
                            second_check_right_y.append(iy)
                            found = True
                            break
                    if not found:
                        continue
            if not second_check_widths:
                new_width.append(stream_width[i])
                new_left_x.append(fix_left_x[i])
                new_left_y.append(fix_left_y[i])
                new_right_x.append(fix_right_x[i])
                new_right_y.append(fix_right_y[i])
                continue
            min_second_width = min(second_check_widths)
            
            min_second_width_ind = np.argmin(second_check_widths)
            new_width.append(min_second_width)
            new_left_x.append(second_check_left_x[min_second_width_ind])
            new_left_y.append(second_check_left_y[min_second_width_ind])
            new_right_x.append(second_check_right_x[min_second_width_ind])
            new_right_y.append(second_check_right_y[min_second_width_ind])
            
    new_width.append(stream_width[-1])
    new_left_x.append(fix_left_x[-1])
    new_left_y.append(fix_left_y[-1])
    new_right_x.append(fix_right_x[-1])
    new_right_y.append(fix_right_y[-1])
    new_mid_x, new_mid_y = [], []
    
    for c in range(len(new_width)):
        new_mid_x.append((new_left_x[c] + new_right_x[c]) / 2)
        new_mid_y.append((new_left_y[c] + new_right_y[c]) / 2)
    return new_mid_x, new_mid_y, new_width, new_left_x, new_left_y, new_right_x, new_right_y
        
#%% This cell deals with Q-D curve using Manning Eqn.
def getQDCurveDataUsingTrapezoid(width, side_slope_1, side_slope_2, slope, manning, d_min, d_max, d_interval):
    n = int((d_max - d_min) // d_interval) + 1
    Q_res, D_res = [], []
    for i in range(n):
        curr_d = d_min + i * d_interval
        f_b = width - (side_slope_1 + side_slope_2) * curr_d + curr_d * np.sqrt(1 + side_slope_1 ** 2) + curr_d * np.sqrt(1 + side_slope_2 ** 2)
        f_t = (width - (side_slope_1 + side_slope_2) * curr_d / 2) * curr_d
        v = ((f_t / f_b) ** (2 / 3)) * (slope ** (1 / 2)) / manning
        if f_t < 0:
            break
        # print([f_b, f_t, v])
        Q = (width - (side_slope_1 + side_slope_2) * curr_d / 2) * curr_d * v
        
        Q_res.append(Q)
        D_res.append(curr_d)
    return Q_res, D_res

def getQDCurveDataUsingTriangular(width, side_slope_1, side_slope_2, slope, manning, d_min, d_max, d_interval):
    # Q = A*(A/P)**(2/3)*(S**(1/2))/n
    n = int((d_max - d_min) // d_interval) + 1
    Q_res, D_res = [], []
    for i in range(n):
        curr_d = d_min + i * d_interval
        A = curr_d * ((side_slope_1 + side_slope_2) * curr_d) / 2
        P = curr_d * np.sqrt(1 + side_slope_1 ** 2) + curr_d * np.sqrt(1 + side_slope_2 ** 2)
        
        v = ((A / P) ** (2 / 3)) * (slope ** (1 / 2)) / manning
        
        Q = A * v
        
        Q_res.append(Q)
        D_res.append(curr_d)
    return Q_res, D_res

def getQDCurveUsingRectangular(width, side_slope_1, side_slope_2, slope, manning, d_min, d_max, d_interval):
    n = int((d_max - d_min) // d_interval) + 1
    Q_res, D_res = [], []
    for i in range(n):
        curr_d = d_min + i * d_interval
        A = width * curr_d
        P = curr_d * 2 + width
        v = ((A / P) ** (2 / 3)) * (slope ** (1 / 2)) / manning
        
        Q = A * v
        
        Q_res.append(Q)
        D_res.append(curr_d)
    return Q_res, D_res
        
def getQDCurveUsingParabola(width, side_slope_1, side_slope_2, slope, manning, d_min, d_max, d_interval):
    n = int((d_max - d_min) // d_interval) + 1
    Q_res, D_res = [], []
    for i in range(n):
        curr_d = d_min + i * d_interval
        A = curr_d * width * 2 / 3
        P = width + (8 * curr_d ** 2) / (3 * width)
        v = ((A / P) ** (2 / 3)) * (slope ** (1 / 2)) / manning
        
        Q = A * v
        
        Q_res.append(Q)
        D_res.append(curr_d)
    return Q_res, D_res

#%% This cell compute V using Q = VA or Manning
def getManningTrapezoidV(width, depth, side_slope_1, side_slope_2, slope, manning):
    f_b = width - (side_slope_1 + side_slope_2) * depth + depth * np.sqrt(1 + side_slope_1 ** 2) + depth * np.sqrt(1 + side_slope_2 ** 2)
    f_t = (width - (side_slope_1 + side_slope_2) * depth / 2) * depth
    v = ((f_t / f_b) ** (2 / 3)) * (slope ** (1 / 2)) / manning
    return v

def getManningRectangularV(width, depth, side_slope_1, side_slope_2, slope, manning):
    A = depth * ((side_slope_1 + side_slope_2) * depth) / 2
    P = depth * np.sqrt(1 + side_slope_1 ** 2) + depth * np.sqrt(1 + side_slope_2 ** 2)
    
    v = ((A / P) ** (2 / 3)) * (slope ** (1 / 2)) / manning
    return v

def getManningTriangularV(width, depth, side_slope_1, side_slope_2, slope, manning):
    A = width * depth
    P = depth * 2 + width
    v = ((A / P) ** (2 / 3)) * (slope ** (1 / 2)) / manning
    return v

def getManningParabolaV(width, depth, side_slope_1, side_slope_2, slope, manning):
    A = depth * width * 2 / 3
    P = width + (8 * depth ** 2) / (3 * width)
    v = ((A / P) ** (2 / 3)) * (slope ** (1 / 2)) / manning
    return v

def getTrapezoidV(width, depth, side_slope_1, side_slope_2, Q):
    A = (width - (side_slope_1 + side_slope_2) * depth / 2) * depth
    return Q / A

def getRectangularV(width, depth, side_slope_1, side_slope_2, Q):
    A = width * depth
    return Q / A

def getTriangularV(width, depth, side_slope_1, side_slope_2, Q):
    A = depth * ((side_slope_1 + side_slope_2) * depth) / 2
    return Q / A

def getParabolaV(width, depth, side_slope_1, side_slope_2, Q):
    A = depth * width * 2 / 3
    return Q / A

#%% This cell deals with GT depth computation and GT error computation.
def getGTBottom(GT_depth, GT_dist, surface_elev, center_dist_array):
    GT_bottom = []
    for i in range(len(GT_depth)):
        curr_dist, curr_depth = GT_dist[i], GT_depth[i]
        higher_val, lower_val = find_upper_lower_val(center_dist_array, curr_dist)
        ind_1, val_1 = find_nearest(center_dist_array, higher_val)
        ind_2, val_2 = find_nearest(center_dist_array, lower_val)
        elev_1, elev_2 = surface_elev[ind_1], surface_elev[ind_2]
        if ind_1 == ind_2:
            GT_elev = elev_1
        else:
            k = (elev_1 - elev_2) / (val_1 - val_2)
            GT_elev = elev_2 + (curr_dist - val_2) * k
        GT_bottom.append(GT_elev - GT_depth[i])
    return GT_bottom

def InterpolateBedProfile(GT_bottom, GT_dist, center_dist_array):
    res = []
    start_GT, end_GT = GT_dist[0], GT_dist[-1]   
    point_num = len(center_dist_array)    
    k = 1    
    for j in range(point_num):
        center_dist = center_dist_array[j]
        if center_dist < start_GT:
            slope = (GT_bottom[2] - GT_bottom[0]) / (GT_dist[2] - GT_dist[0])
            temp_bed = GT_bottom[0] + slope * (center_dist - start_GT)
        elif center_dist > end_GT:
            slope = (GT_bottom[-3] - GT_bottom[-1]) / (GT_dist[-3] - GT_dist[-1])
            temp_bed = GT_bottom[-1] + slope * (center_dist - end_GT)
        else:
            if center_dist > GT_dist[k]:
                k = k + 1
            dist_1, dist_2, elev_1, elev_2 = GT_dist[k - 1], GT_dist[k], GT_bottom[k - 1], GT_bottom[k]
            slope = (elev_1 - elev_2) / (dist_1 - dist_2)                
            temp_bed = elev_1 + slope * (center_dist - dist_1)
        res.append(temp_bed)
    return res

#%% This cell deals with extracting required stream segment from the whole delineation.        
def extractRequiredStreamSegment(arr, start_ind, end_ind):
    return arr[start_ind : end_ind + 1]

def findIndForAPoint(x_arr, y_arr, x_point, y_point):
    arr_len = len(x_arr)
    
    for i in range(arr_len):
        temp_x = x_arr[i]
        temp_y = y_arr[i]
        temp_dist = DistanceBetweenTwoPoints(temp_x, temp_y, x_point, y_point)
        if i == 0:
            min_dist, min_ind = temp_dist, 0
        else:
            if temp_dist < min_dist:
                min_dist, min_ind = temp_dist, i
    return min_ind, min_dist
                
#%% This cell deals with img masking for white pixel counting
def getBWImg(img):
    bw0 = np.where(img[:,:,0] >= 252, 1, 0)
    bw1 = np.where(img[:,:,1] >= 252, 1, 0)
    bw2 = np.where(img[:,:,2] >= 252, 1, 0)
    bw = np.where((bw0 + bw1 + bw2) == 3, 1, 0)
    return bw

def getExtractCoords(CX, CY, LX, LY, RX, RY, scale, extend):
    delta_x = int((CX - extend[0]) / extend[1])
    delta_y = int((CY - extend[3]) / extend[5])
    dx = RX - LX
    dy = RY - LY
    if dx == 0:
        ex, ey = 0, 1
    elif dy == 0:
        ex, ey = 1, 0
    else:
        ex = abs(dx / np.sqrt(dx ** 2 + dy ** 2))
        ey = abs(dy / np.sqrt(dx ** 2 + dy ** 2))
    first_move = 1 / np.sqrt(extend[1] ** 2 +  extend[5] ** 2)
    second_move = 0.5 / np.sqrt(extend[1] ** 2 +  extend[5] ** 2)
    first_delta_x1, first_delta_y1 = ex * first_move * dx / abs(dx), ey * first_move * dy / abs(dy)
    first_delta_x2, first_delta_y2 = - first_delta_x1, - first_delta_y1
    x3, y3, x4, y4 = delta_x + first_delta_x1, delta_y + first_delta_y1, delta_x + first_delta_x2, delta_y +  first_delta_y2
    ex2, ey2 = ey, ex
    if dx == 0 or dy == 0:
        dy1, dx1 = dx, dy
    else:
        dy1, dx1 = -dx, dy
    
    second_delta_x1, second_delta_y1 = ex2 * second_move * dx1 / abs(dx1), ey2 * second_move * dy1 / abs(dy1)
    second_delta_x2, second_delta_y2 = - second_delta_x1, - second_delta_y1
    p1_x, p1_y, p2_x, p2_y, p3_x, p3_y, p4_x, p4_y = x3 + second_delta_x1, y3 + second_delta_y1,\
          x3 + second_delta_x2, y3 + second_delta_y2, x4 + second_delta_x1, y4 + second_delta_y1,\
                x4 + second_delta_x2, y4 + second_delta_y2
    d1, d2 = DistanceBetweenTwoPoints(p2_x, p2_y, p3_x, p3_y), DistanceBetweenTwoPoints(p2_x, p2_y, p4_x, p4_y)
    p1_x, p1_y, p2_x, p2_y, p3_x, p3_y, p4_x, p4_y = int(p1_x / scale), int(p1_y / scale), int(p2_x / scale), int(p2_y / scale),\
                                                     int(p3_x / scale), int(p3_y / scale), int(p4_x / scale), int(p4_y / scale)
    if d1 >= d2:
        return [[p1_x, p1_y], [p2_x, p2_y], [p4_x, p4_y], [p3_x, p3_y]]
    else:
        return [[p1_x, p1_y], [p2_x, p2_y], [p3_x, p3_y], [p4_x, p4_y]]
    
def getWhitePixelRatioWithoutEyot(CX, CY, LX, LY, RX, RY, scale, extend, BW_img):
    row_num, col_num = BW_img.shape[:2]
    points = getExtractCoords(CX, CY, LX, LY, RX, RY, scale, extend)
    orig_img = np.zeros((row_num, col_num), np.uint8)
    cv2.fillConvexPoly(orig_img, np.array(points, 'int32'), 1)
    white_img = orig_img + BW_img
    white_ratio = np.sum(np.where(white_img == 2, 1, 0)) / np.sum(white_img)
    return white_ratio
    
def getPolygonCoordsWithEyot(LX, LY, LIX, LIY, RX, RY, RIX, RIY, scale, extend, BW_img):
    row_num, col_num = BW_img.shape[:2]
    CX1, CY1 = (LIX + LX) / 2, (LIY + LY) / 2
    points_1 = getExtractCoords(CX1, CY1, LX, LY, LIX, LIY, scale, extend)
    CX2, CY2 = (RIX + RX) / 2, (RIY + RY) / 2
    points_2 = getExtractCoords(CX2, CY2, RIX, RIY, RX, RY, scale, extend)
    
    orig_img = np.zeros((row_num, col_num), np.uint8)
    cv2.fillConvexPoly(orig_img, np.array(points_1, 'int32'), 1)
    cv2.fillConvexPoly(orig_img, np.array(points_2, 'int32'), 1)
    white_img = orig_img + BW_img
    white_ratio = np.sum(np.where(white_img == 2, 1, 0)) / np.sum(white_img)
    return white_ratio

#%% This cell computes length-related information.
def computeWholeDistance(x_list, y_list):
    s_x = x_list[0 : -1]
    t_x = x_list[1:]
    s_y = y_list[0 : -1]
    t_y = y_list[1:]
    x_sq = np.power(np.subtract(s_x, t_x), 2)
    y_sq = np.power(np.subtract(s_y, t_y), 2)
    res = np.sum(np.sqrt(x_sq + y_sq))
    return res

def getLengthArray(x_list, y_list):
    res = []
    point_num = len(x_list)
    for i in range(point_num):
        if i == 0:
            res.append(0)
        else:
            res.append(res[-1] + DistanceBetweenTwoPoints(x_list[i], y_list[i], x_list[i - 1], y_list[i - 1]))
    return res
        
#%% This cell deals with the bank top determination

def NewcheckTopForOneSide(trans_dist, trans_elev, start_ind, end_ind, bank_elev, bank_dist, interval, check_length, angle_thresh):
    end_dist, end_elev = trans_dist[end_ind], trans_elev[end_ind]
    # print([bank_elev, bank_dist, end_dist, end_elev, check_length])
    # print([start_ind, end_ind])
    cos_thresh = math.cos(math.pi * angle_thresh / 180)
    for p in range(start_ind, end_ind, interval):
        curr_dist, curr_elev = trans_dist[p], trans_elev[p]
        dx_1, dy_1 = abs(curr_dist - bank_dist), curr_elev - bank_elev
        check_dist_0 = curr_dist + check_length
        if abs(check_dist_0 - bank_dist) >= abs(end_dist - bank_dist):
            break
        else:
            check_ind, check_dist = find_nearest(trans_dist, check_dist_0)
            check_elev = trans_elev[check_ind]
            dx_2, dy_2 = check_dist - bank_dist, check_elev - bank_elev
            cos_theta = (dx_1 * dx_2 + dy_1 * dy_2) / (np.sqrt(dx_1 ** 2 + dy_1 ** 2) * np.sqrt(dx_2 ** 2 + dy_2 ** 2))
            # print([cos_thresh, cos_theta, dx_1, dx_2, dy_1, dy_2, curr_dist, check_dist_0, check_elev, p])
            if abs(cos_theta) < abs(cos_thresh) and dy_2 / dx_2 < dy_1 / dx_1:
                break
    return p, trans_dist[p], trans_elev[p] 
            
def getBankTop(trans_dist, trans_elev, surface_elev, left_bank_elev, left_bank_dist, right_bank_elev, \
                           right_bank_dist, left_interval = -1, right_interval = 1,\
                           left_check_length = -5, right_check_length = 5, angle_thresh = 30):
    left_bank_ind, left_bank_dist = find_nearest(trans_dist, left_bank_dist)
    right_bank_ind, right_bank_dist = find_nearest(trans_dist, right_bank_dist)
    
    

    left_dist1, right_dist1 = 5, trans_dist[-1] - 5
    
   
    
    left_end_ind, left_end_dist = find_nearest(trans_dist, left_dist1)
    right_end_ind, right_end_dist = find_nearest(trans_dist, right_dist1)
    
    left_ind = FindClosestNoLessThanInd(trans_elev, surface_elev + 0.3, left_bank_ind + 20 * left_interval, \
                             0, left_interval)
        
    right_ind = FindClosestNoLessThanInd(trans_elev, surface_elev + 0.3, right_bank_ind + 20 * right_interval, \
                             0, right_interval)
    # print([left_ind, right_ind, trans_dist[left_ind], trans_dist[right_ind]])
    
    left_top_ind, left_top_dist, left_top_elev = \
        NewcheckTopForOneSide(trans_dist, trans_elev, left_ind, left_end_ind, left_bank_elev, \
                           left_bank_dist, left_interval, left_check_length, angle_thresh)
    # print("=======================================================")
    right_top_ind, right_top_dist, right_top_elev = \
        NewcheckTopForOneSide(trans_dist, trans_elev, right_ind, right_end_ind, right_bank_elev, \
                           right_bank_dist, right_interval, right_check_length, angle_thresh)
    return left_top_ind, left_top_dist, left_top_elev, right_top_ind, right_top_dist, right_top_elev



def ModifySurfaceElevation(trans_elev, left_bank_ind, right_bank_ind, surface_elev):
    point_num = len(trans_elev)
    res = deepcopy(trans_elev)
    for i in range(point_num):
        if i >= min(left_bank_ind, right_bank_ind) and i <= max(left_bank_ind, right_bank_ind):
            res[i] = surface_elev
    for j in range(left_bank_ind, -1, -1):
        if res[j] <= surface_elev:
            res[j] = surface_elev
        elif res[j] >= surface_elev + 1:
            break
    for k in range(right_bank_ind, point_num):
        if res[k] <= surface_elev:
            res[k] = surface_elev
        elif res[k] >= surface_elev + 1:
            break
        
    return res

def computeSaturation(width, trans_dist, trans_elev, check_left_elev, check_right_elev, surface_elev):
    # print(check_left_elev, check_right_elev, surface_elev)
    check_elev = min(check_left_elev, check_right_elev)
    mid_ind = int(len(trans_dist) // 2)
    left_dist = findExactVal(trans_dist, trans_elev, check_elev, mid_ind, 0, -1)
    right_dist = findExactVal(trans_dist, trans_elev, check_elev, mid_ind, len(trans_dist) - 1, 1)
    # print([left_dist, right_dist])
    return width / (abs(left_dist - right_dist))

def findExactVal(vals, refs, check_ref, start_ind, end_ind, interval):
    res_1 = []
    # print([start_ind, end_ind, interval])
    for i in range(start_ind, end_ind, interval):
        if i == end_ind:
            continue
        else:
            curr_res_1 = refs[i]
            curr_res_2 = refs[i + interval]
            #print([curr_res_1, curr_res_2, check_ref])
            if (curr_res_1 - check_ref) * (curr_res_2 - check_ref) <= 0:
                res_1.append(i)
    # print(res_1)
    if not res_1:
        res_ind = start_ind
    else:   
        res_ind = min(res_1) if interval > 0 else max(res_1)
    val_1, val_2, ref_1, ref_2 = vals[res_ind], vals[res_ind + interval], refs[res_ind], refs[res_ind + interval]
    weight = (check_ref - ref_1) / (ref_2 - ref_1)
    final_val = val_1 + weight * (val_2 - val_1)
    return final_val

def FindClosestNoLessThanInd(refs, check_ref, start_ind, end_ind, interval):
    res = start_ind
    for i in range(start_ind, end_ind, interval):
        if i == end_ind:
            continue
        else:
            curr_res_1 = refs[i]
            curr_res_2 = refs[i + interval]
            if (curr_res_1 - check_ref) * (curr_res_2 - check_ref) <= 0:
                res = i + interval
                break
    return res
    
#%% This cell computes eyot-related information.
def CrossOfTwoVectors(p1, p2, p3, p4):
    x1 = p2[0] - p1[0]
    x2 = p4[0] - p3[0]
    y1 = p2[1] - p1[1]
    y2 = p4[1] - p3[1]
    return np.cross(np.array([x1, y1]), np.array([x2, y2]))

def RectangularTestOfIntersection(p1, p2, p3, p4):
    l_x_min, l_x_max = min(p1[0], p2[0]), max(p1[0], p2[0])
    r_x_min, r_x_max = min(p3[0], p4[0]), max(p3[0], p4[0])
    l_y_min, l_y_max = min(p1[1], p2[1]), max(p1[1], p2[1])
    r_y_min, r_y_max = min(p3[1], p4[1]), max(p3[1], p4[1])
    return l_x_max >= r_x_min and r_x_max >= l_x_min \
        and r_y_max >= l_y_min and l_y_max >= r_y_min 

def IsTwoSegmentIntersected(p1, p2, p3, p4):
    if RectangularTestOfIntersection(p1, p2, p3, p4):
        if np.dot(CrossOfTwoVectors(p3, p1, p3, p4), \
                  CrossOfTwoVectors(p3, p2, p3, p4)) <= 0\
        and np.dot(CrossOfTwoVectors(p2, p3, p2, p1), \
                  CrossOfTwoVectors(p2, p4, p2, p1)) <= 0:
            return True
        else:
            return False
    else:
        return False

def computeIntersectPoint(p1, p2, p3, p4):
    x1, y1, x2, y2 = p1[0], p1[1], p2[0], p2[1]
    x3, y3, x4, y4 = p3[0], p3[1], p4[0], p4[1]
    a0, b0, c0 = y1 - y2, x2 - x1, x1 * y2 - x2 * y1
    a1, b1, c1 = y3 - y4, x4 - x3, x3 * y4 - x4 * y3
    d = a0 * b1 - a1 * b0
    if d == 0:
        return [-999999, -999999]
    else:
        x = (b0 * c1 - b1 * c0) / d
        y = (c0 * a1 - c1 * a0) / d
        return [x, y]



#%% This cell computes orientation-related information.
def subtractChannelPointByBoundary(x_list, y_list, lt_point):
    return x_list - lt_point[0], y_list - lt_point[1]

def createChannelImg(new_left_x, new_left_y, new_right_x, new_right_y, lt_point, rb_point):
    row_num = int(abs(lt_point[0] - rb_point[0])) + 1
    col_num = int(abs(lt_point[1] - rb_point[1])) + 1
    empty_img = np.zeros((row_num, col_num, 3), np.unit8)
    all_row = np.array(new_left_x + new_right_x[::-1])
    all_col = np.array(new_left_y + new_right_y[::-1])
    pts = np.transpose(np.vstack((all_row, all_col)))
    cv2.fillPoly(empty_img, [pts], (255, 0, 0))
    return empty_img

def computeChannelOrientation(channel_img):
    gray_img = channel_img[:, :, 0]
    ret, thresh_img = cv2.threshold(gray_img, 0, 255, 0)
    row_num, col_num = thresh_img.shape[:2]
    
    white_locs = np.where(thresh_img > 0)
    temp_Xset = white_locs[1]
    temp_Yset = white_locs[0]
    area = temp_Yset.shape[0]
    
    x_center = sum(temp_Xset) / area
    y_center = sum(temp_Yset) / area
    
    temp_a = np.multiply(temp_Xset - x_center, temp_Xset - x_center)
    temp_b = 2 * np.multiply(temp_Xset - x_center, temp_Yset - y_center)
    temp_c = np.multiply(temp_Yset - y_center, temp_Yset - y_center)
        
    a, b, c = sum(temp_a), sum(temp_b), sum(temp_c)
    if a == c and b == 0:
        orientation = -100000
    elif a == c:
        orientation = 0.25 * math.pi
    elif b == 0:
        orientation = 0
    else:
        orientation = math.atan2(b, a - c) / 2
        
    if a + c + math.sqrt( b * b + (a-c) * (a-c)) == 0:
        roundness = -100000
    else:
        roundness = (a + c - math.sqrt( b * b + (a-c) * (a-c))) / (a + c + math.sqrt( b * b + (a-c) * (a-c)))
         
    min_itia = (a + c - math.sqrt( b * b + (a-c) * (a-c)))
    res = [area, orientation, roundness, x_center, y_center, min_itia]
    return res

#%% This cell deals with bend computation
def computeDirectionOfEachPoint(mid_x_list, mid_y_list, start_ind, end_ind):
    # start_ind should be at least 5, end ind should be at most length - 4, the suggestion is to make other direction vals as np.nan
    
    thetas = []
    for i in range(start_ind, end_ind):
        x1 = mid_x_list[i - 4]
        y1 = mid_y_list[i - 4]
        x2 = mid_x_list[i + 4]
        y2 = mid_y_list[i + 4]
        if y1 == y2:
            theta = 0 if x2 >= x1 else 180
        elif x1 == x2:
            theta = 90 if y2 >= y1 else 270
        else:
            tan_val = (y2 - y1) / (x2 - x1)
            orig_angle = 180 * math.atan(tan_val) / math.pi
            if orig_angle > 0:
                theta = orig_angle if y2 > y1 else 180 + orig_angle
            else:
                theta = 180 - orig_angle if y2 > y1 else 360 - orig_angle
        thetas.append(theta)
    return thetas

def computeAngleBetweenTwoThetas(theta1, theta2):
    # Herein, positive means left turn, negative means right turn.
    orig_turning = theta2 - theta1
    if orig_turning > 180:
        turning = orig_turning - 360
    elif orig_turning < -180:
        turning = orig_turning + 360
    else:
        turning = orig_turning
    return turning
    

def computeTurnings(direction, interval):
    dir_num = len(direction)
    res = [np.nan]
    if interval % 2 == 0:
        inter_1, inter_2 = round(interval / 2), round(interval / 2)
    else:
        inter_1, inter_2 = int(interval / 2), int(interval / 2) + 1
    for i in range(inter_1, dir_num - inter_2):
        res.append(computeAngleBetweenTwoThetas(direction[i - inter_1], direction[i + inter_2]))    
    return res, inter_1, inter_2

def computeBendLocations(mid_x, mid_y, bend_ind_1, bend_ind_2, slope_interval, angle_thresh):   
    direction_res = computeDirectionOfEachPoint(mid_x, mid_y, bend_ind_1, bend_ind_2)
    turnings, inter_1, inter_2 = computeTurnings(direction_res, slope_interval)
    Cat = [1 for b in range(bend_ind_1)] + [1 for b in range(inter_1)]
    for t in range(len(turnings)):
        if abs(turnings[t]) > angle_thresh:
            Cat.append(2)
            #print(t + bend_ind_1)
        else:
            Cat.append(1)
    Cat = Cat + [1 for b in range(inter_2)] + [1 for b in range(bend_ind_2 - 1, len(mid_x))]
    cat_ind1, cat_ind2, curr_val, prev_val = [], [], 1, 1
    for i in range(len(Cat)):
        curr_val = Cat[i]
        if curr_val == 2 and prev_val != 2:
            cat_ind1.append(i)
        elif curr_val != 2 and prev_val == 2:
            cat_ind2.append(i - 1)
        prev_val = Cat[i]
    for c in range(len(cat_ind1)):
        if cat_ind1[c] == cat_ind2[c]:
            Cat[cat_ind1[c]] = 1
        else:
            for ci in range(cat_ind1[c], cat_ind2[c] + 1):
                if ci != round((cat_ind1[c] + cat_ind2[c]) / 2):
                    # print(round((cat_ind1[c] + cat_ind2[c]) / 2))
                    Cat[ci] = 1
    
    cat_ind1, cat_ind2, curr_val, prev_val = [], [], 1, 1
    for i in range(len(Cat)):
        curr_val = Cat[i]
        if curr_val == 2 and prev_val != 2:
            cat_ind1.append(i)
        elif curr_val != 2 and prev_val == 2:
            cat_ind2.append(i - 1)
        prev_val = Cat[i]
    return Cat, cat_ind1, cat_ind2
    
#%% This cell computes depth-related information.    
def EstimateDepthForASingleTransect(w, lz, rz, s, n, Q, y_max, thresh):
    y_num = int(y_max / 0.01)
    y_vals = np.linspace(0, y_max, y_num)
    smallest_ind = 0
    is_found = False
    for i in range(y_num):
        y = y_vals[i]
        A = w * y - y * y * (lz + rz) / 2    
        P = 2 * w - (lz + rz) * y + y * np.sqrt(1 + lz ** 2) + y * np.sqrt(1 + rz ** 2)
        v = (1 / n) * (A / P) ** (2 / 3) * s ** (1 / 2)
        delta_Q = abs(Q - A * v)
        if delta_Q <= thresh:
            is_found = True
            break
        elif i == 0:
            smallest_Q = delta_Q
        elif delta_Q <= smallest_Q:
            smallest_Q = delta_Q
            smallest_ind = i
    if is_found:
        return y_vals[i]
    else:
        return y_vals[smallest_ind]

#%% This cell deals with geographic system coversion.
def getSRSPair(dataset):
    prosrs = osr.SpatialReference()
    prosrs.ImportFromWkt(dataset.GetProjection())
    geosrs = prosrs.CloneGeogCS()
    return prosrs, geosrs
 
def geo2lonlat(dataset, x, y):
    prosrs, geosrs = getSRSPair(dataset)
    ct = osr.CoordinateTransformation(prosrs, geosrs)
    coords = ct.TransformPoint(x, y)
    return coords[:2]
 
 
def lonlat2geo(dataset, lon, lat):
    prosrs, geosrs = getSRSPair(dataset)
    ct = osr.CoordinateTransformation(geosrs, prosrs)
    coords = ct.TransformPoint(lon, lat)
    return coords[:2]
 
def imagexy2geo(dataset, row, col):
    trans = dataset.GetGeoTransform()
    px = trans[0] + col * trans[1] + row * trans[2]
    py = trans[3] + col * trans[4] + row * trans[5]
    return px, py
 
 
def geo2imagexy(dataset, x, y):
    trans = dataset.GetGeoTransform()
    a = np.array([[trans[1], trans[2]], [trans[4], trans[5]]])
    b = np.array([x - trans[0], y - trans[3]])
    return np.linalg.solve(a, b) 

def get_img_from_raster(raster_folder, raster_file):    
    ds = gdal.Open(raster_folder + raster_file)
    myarray1 = np.array(ds.GetRasterBand(1).ReadAsArray())
    myarray2 = np.array(ds.GetRasterBand(2).ReadAsArray())
    myarray3 = np.array(ds.GetRasterBand(3).ReadAsArray())
    return np.stack((np.uint8(myarray3), np.uint8(myarray2),np.uint8(myarray1)), axis = 2)

def get_tif_info(tif_path):
    if tif_path.endswith('.tif') or tif_path.endswith('.TIF'):
        dataset = gdal.Open(tif_path)
        pcs = osr.SpatialReference()
        pcs.ImportFromWkt(dataset.GetProjection())
        gcs = pcs.CloneGeogCS()
        extend = dataset.GetGeoTransform()
        # im_width = dataset.RasterXSize 
        # im_height = dataset.RasterYSize 
        shape = (dataset.RasterYSize, dataset.RasterXSize)
    else:
        raise "Unsupported file format"

    img = dataset.GetRasterBand(1).ReadAsArray()  # (height, width)
    return img, dataset, gcs, pcs, extend, shape

def lonlat_to_xy(gcs, pcs, lon, lat):
    ct = osr.CoordinateTransformation(gcs, pcs)
    coordinates = ct.TransformPoint(lon, lat)
    return coordinates[0], coordinates[1], coordinates[2]


def xy_to_lonlat(gcs, pcs, x, y):
    ct = osr.CoordinateTransformation(gcs, pcs)
    lon, lat, _ = ct.TransformPoint(x, y)
    return lon, lat


def xy_to_rowcol(extend, x, y):
    # a = np.array([[extend[1], extend[2]], [extend[4], extend[5]]])
    # b = np.array([x - extend[0], y - extend[3]])

    # row_col = np.linalg.solve(a, b)
    # row = int(np.floor(row_col[1]))
    # col = int(np.floor(row_col[0]))
    col = int((x - extend[0]) / extend[1])
    row = int((y - extend[3]) / extend[5])
    return row, col


def rowcol_to_xy(extend, row, col):
    x = extend[0] + col * extend[1] + row * extend[2]
    y = extend[3] + col * extend[4] + row * extend[5]
    return x, y

def get_value_by_coordinates(img, dataset, gcs, pcs, extend, shape, coordinates, coordinate_type='rowcol'):
    
    if coordinate_type == 'rowcol':
        value = img[coordinates[0], coordinates[1]]
    elif coordinate_type == 'lonlat':
        x, y, _ = lonlat_to_xy(gcs, pcs, coordinates[0], coordinates[1])
        row, col = xy_to_rowcol(extend, x, y)
        if row >= shape[0] or col >= shape[1]:
            value = np.nan
        elif row < 0 or col < 0:
            value = np.nan
        else:
            value = img[row, col]
    elif coordinate_type == 'xy':
        row, col = xy_to_rowcol(extend, coordinates[0], coordinates[1])
        if row >= shape[0] or col >= shape[1]:
            value = np.nan
        elif row < 0 or col < 0:
            value = np.nan
        else:
            value = img[row, col]
    else:
        raise 'coordinated_type error'
    return value

#%% This cell designed for detecting bends
def computeAnglesBetweenTwoVectors(x, y):
    x1, y1, x2, y2 = x[0], x[1], y[0], y[1]
    mode_1 = np.sqrt(x1 ** 2 + y1 ** 2)
    mode_2 = np.sqrt(x2 ** 2 + y2 ** 2)
    dot_val = x1 * x2 + y1 * y2
    cos_val = dot_val / (mode_1 * mode_2)
    angle = np.arccos(cos_val) * 180 / np.pi
    return angle

def detectBends(x_list, y_list, thresh):
    point_num = len(x_list)
    x_start = x_list[0 : point_num - 1]
    x_end = x_list[2 : point_num]
    y_start = y_list[0 : point_num - 1]
    y_end = y_list[2 : point_num]
    x_delta = x_end - x_start
    y_delta = y_end - y_start
    angles = []
    for i in range(point_num - 1):
        angles.append(computeAnglesBetweenTwoVectors(x_delta[i], y_delta[i]))
    a_start = angles[0 : point_num - 2]
    a_end = angles[1 : point_num - 1]
    a_delta = np.array(abs(a_end - a_start))
    return list(np.where(a_delta >= thresh)[0] + 1)

def computeSinuosity(x_list, y_list):
    direct_dist = DistanceBetweenTwoPoints(x_list[0], y_list[0], x_list[-1], y_list[-1])
    curved_dist = computeWholeDistance(x_list, y_list)
    return curved_dist / direct_dist

#%% This cell designed for functions buffers the transect line
def getTransectLine(left_x_list, left_y_list, right_x_list, right_y_list, ind):
    lx, ly, rx, ry = left_x_list[ind], left_y_list[ind], right_x_list[ind], right_y_list[ind]
    A, B, C = getABCofTwoPointSegment(lx, ly, rx, ry)
    return A, B, C

def getLeftAndRightBufferLine(A, B, C, delta):
    if A == 0 or B == 0:
        return [A, B, C - delta], [A, B, C + delta]
    else:
        d_k = np.sqrt(1 + (1 / A) ** 2) * delta
        return [A, B, C - d_k], [A, B, C + d_k]

def getABCofTwoPointSegment(lx, ly, rx, ry):
    if lx == rx:
        A, B, C = 1, 0, -lx
    elif ly == ry:
        A, B, C = 0, 1, -ly
    else:
        A, B, C = (ry - ly) / (rx - lx), -1, (rx * ly - lx * ry) / (rx - lx)
    return A, B, C

def getIntersectsOfTwoABCs(A1, B1, C1, A2, B2, C2):
    if A1 == 0 or A2 == 0 or B1 == 0 or B2 == 0:
        if (A1 == 0 and A2 == 0) or (B1 == 0 and B2 == 0):
            x, y = np.NaN, np.NaN
        elif A1 == 0:
            y = - C1 / B1
            x = (- C2 - B2 * y) / A2
        elif A2 ==0:
            y = - C2 / B2           
            x = (- C1 - B1 * y) / A1
        elif B1 == 0:
            x = - C1 / A1
            y = (- C2 - A2 * x) / B2
        elif B2 == 0:
            x = - C2 / A2
            y = (- C1 - A1 * x) / B1
    elif A1 / B1 == A2 /B2:
        x, y = np.NaN, np.NaN
    else:
        x = (B2 * C1 - B1 * C2) / (B1 * A2 - A1 * B2)
        y = (A2 * C1 - A1 * C2) / (B2 * A1 - A2 * B1)
    return x, y    

def getLeftAndRightBufferLineIntersection(left_x_list, left_y_list, right_x_list,\
        right_y_list, left_coef, right_coef, ind, delta, total_dist):
    point_num = len(left_x_list)
    est_interval = int(np.ceil(point_num * total_dist / delta))
    min_ind = max(0, ind - est_interval) 
    max_ind = min(ind + est_interval, point_num - 1)
    l_l_found, l_r_found, r_l_found, r_r_found = False, False, False, False
    A_l, B_l, C_l = left_coef[0], left_coef[1], left_coef[2]
    A_r, B_r, C_r = right_coef[0], right_coef[1], right_coef[2]
    
    for i in range(min_ind, max_ind):
        xl_p1, xl_p2, xr_p1, xr_p2 = left_x_list[i], left_x_list[i + 1],\
                                 right_x_list[i], right_x_list[i + 1]
        yl_p1, yl_p2, yr_p1, yr_p2 = left_y_list[i], left_y_list[i + 1],\
                                 right_y_list[i], right_y_list[i + 1]
        if not l_l_found:
            l_l_found = (A_l * xl_p1 + B_l * yl_p1 + C_l) * \
                (A_l * xl_p2 + B_l * yl_p2 + C_l) <= 0
            if l_l_found:
                A, B, C = getABCofTwoPointSegment(xl_p1, yl_p1, xl_p2, yl_p2)
                ll_x, ll_y = getIntersectsOfTwoABCs(A, B, C, A_l, B_l, C_l)
                ll_ind = i
        if not l_r_found:
            l_r_found = (A_r * xl_p1 + B_r * yl_p1 + C_r) * \
                (A_r * xl_p2 + B_r * yl_p2 + C_r) <= 0
            if l_r_found:
                A, B, C = getABCofTwoPointSegment(xl_p1, yl_p1, xl_p2, yl_p2)
                lr_x, lr_y = getIntersectsOfTwoABCs(A, B, C, A_r, B_r, C_r)
                lr_ind = i
        if not r_l_found:
            r_l_found = (A_l * xr_p1 + B_l * yr_p1 + C_l) * \
                (A_l * xr_p2 + B_l * yr_p2 + C_l) <= 0
            if r_l_found:
                A, B, C = getABCofTwoPointSegment(xr_p1, yr_p1, xr_p2, yr_p2)
                rl_x, rl_y = getIntersectsOfTwoABCs(A, B, C, A_l, B_l, C_l)
                rl_ind = i
        if not r_r_found:
            r_r_found = (A_r * xr_p1 + B_r * yr_p1 + C_r) * \
                (A_r * xr_p2 + B_r * yr_p2 + C_r) <= 0
            if r_l_found:
                A, B, C = getABCofTwoPointSegment(xr_p1, yr_p1, xr_p2, yr_p2)
                rr_x, rr_y = getIntersectsOfTwoABCs(A, B, C, A_r, B_r, C_r)
                rr_ind = i
        if l_l_found and l_r_found and r_l_found and r_r_found:
            break
    if not l_l_found:
        ll_x, ll_y, ll_ind = left_x_list[0], left_y_list[0], 0
    if not l_r_found:
        lr_x, lr_y, lr_ind = left_x_list[point_num - 1], \
            left_y_list[point_num - 1], point_num - 1
    if not r_l_found:
        rl_x, rl_y, rl_ind = right_x_list[0], right_y_list[0], 0
    if not r_r_found:
        rr_x, rr_y, rr_ind = right_x_list[point_num - 1], \
            right_y_list[point_num - 1], point_num - 1
    big_l_ind, small_l_ind = max(ll_ind, lr_ind), min(ll_ind, lr_ind)
    big_r_ind, small_r_ind = max(rl_ind, rr_ind), min(rl_ind, rr_ind)
    ext_lx_list = left_x_list[small_l_ind : big_l_ind + 1]
    ext_ly_list = left_y_list[small_l_ind : big_l_ind + 1]
    ext_rx_list = right_x_list[small_r_ind : big_r_ind + 1]
    ext_ry_list = right_y_list[small_r_ind : big_r_ind + 1]
    l_len = big_l_ind + 1 - small_l_ind
    r_len = big_r_ind + 1 - small_r_ind
    if ext_lx_list[0] != ll_x and ext_ly_list[0] != ll_y:
        ext_lx_list = [ll_x] + ext_lx_list[1:]
        ext_ly_list = [ll_y] + ext_ly_list[1:]
    if ext_rx_list[0] != rl_x and ext_ry_list[0] != rl_y:
        ext_rx_list = [rl_x] + ext_rx_list[1:]
        ext_ry_list = [rl_y] + ext_ry_list[1:]
    if ext_lx_list[l_len - 1] != lr_x and ext_ly_list[l_len - 1] != lr_y:
        ext_lx_list = ext_lx_list[: l_len - 1] + [lr_x]
        ext_ly_list = ext_ly_list[: l_len - 1] + [lr_y]
    if ext_rx_list[r_len - 1] != rr_x and ext_ry_list[r_len - 1] != rr_y:
        ext_rx_list = ext_rx_list[: r_len - 1] + [rr_x]
        ext_ry_list = ext_ry_list[: r_len - 1] + [rr_y]
    return ext_lx_list, ext_ly_list, ext_rx_list, ext_ry_list

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]

def find_upper_lower_val(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    check_val = array[idx]
    if idx == 0:
        check_val_1 = array[idx + 1]
    elif idx == len(array) - 1:
        check_val_1 = array[idx - 1]
    else:
        check_val_1 = max(array[idx + 1], array[idx - 1]) if array[idx + 1] > value and array[idx - 1] > value \
            else min(array[idx + 1], array[idx - 1])
            
    if check_val < value:
        low_val = check_val        
        upper_val = check_val_1 if check_val_1 > value else -999999
    elif check_val == value:
        low_val, upper_val = check_val, check_val        
    else:
        upper_val = check_val
        low_val = check_val_1 if check_val_1 < value else -999999
    return low_val, upper_val

def find_close_vals(array, value, thresh):
    array_len = len(array)
    res, ind = [], []
    for i in range(array_len):
        if abs(array[i] - value) <= thresh:
            res.append(array[i])
            ind.append(i)
    return ind, res

def getWeight(low_val, upper_val, val):
    return (val - low_val) / (upper_val - low_val)

def getInterPolatedCoords(x1, x2, y1, y2, weight):
    x = x1 + weight * (x2 - x1)
    y = y1 + weight * (y2 - y1)
    return x, y
    
def getInterPolatedVals(a1, a2, weight):    
    low_val, upper_val = min(a1, a2), max(a1, a2)
    return low_val + weight * (upper_val - low_val)

#%% This cell deal with generating rays and polygons for each transect
def generateRayForATransect(left_x, left_y, right_x, right_y, extended_len = 15):
    dx = right_x - left_x
    dy = right_y - left_y
    if dx == 0:
        ex, ey = 0, extended_len
    elif dy == 0:
        ex, ey = extended_len, 0
    else:
        ex = extended_len * abs(dx / np.sqrt(dx ** 2 + dy ** 2))
        ey = extended_len * abs(dy / np.sqrt(dx ** 2 + dy ** 2))
    dr_x = ex if dx > 0 else - ex
    dr_y = ey if dy > 0 else - ey
    dl_x, dl_y = - dr_x, - dr_y
    return left_x + dl_x, left_y + dl_y, right_x + dr_x, right_y + dr_y

def generateAllRays(Left_X, Left_Y, Right_X, Right_Y, extended_len = 15):
    point_num = len(Left_X)
    L_X, L_Y, R_X, R_Y = [], [], [], []
    for i in range(point_num):
        x_l, y_l, x_r, y_r = Left_X[i], Left_Y[i], Right_X[i], Right_Y[i]
        left_ray_x, left_ray_y, right_ray_x, right_ray_y = generateRayForATransect(x_l, y_l, x_r, y_r, extended_len = 15)
        L_X.append(left_ray_x)
        L_Y.append(left_ray_y)
        R_X.append(right_ray_x)
        R_Y.append(right_ray_y)
    return L_X, L_Y, R_X, R_Y

def convertFeetUnitToMeterUnit(feet_len):
    return 0.3048 * feet_len

def convertMeterUnitToFeetUnit(meter_len):
    return meter_len / 0.3048

def InterpolateAllPoints(x_1, x_2, y_1, y_2, x_unit, y_unit):
    distance = np.sqrt((x_2 - x_1) ** 2 + (y_2 - y_1) ** 2)
    unit_distance = np.sqrt(x_unit ** 2 + y_unit ** 2)
    point_num = int(distance // unit_distance)
    dx, dy = (x_2 - x_1) / point_num , (y_2 - y_1) / point_num
    x_res, y_res = [], []
    for i in range(point_num):
        x_res.append(x_1 + i * dx)
        y_res.append(y_1 + i * dy)
    
    return x_res, y_res

def ExtractDEMProfileOfATransect(x_1, x_2, y_1, y_2, img, dataset, gcs, pcs, extend, shape):
    
    x_unit, y_unit = extend[1], extend[5]
    point_xs, point_ys = InterpolateAllPoints(x_1, x_2, y_1, y_2, x_unit, y_unit)
    vals, dists = [], []
    # x_start, y_start = extend[0], extend[3]
    for i in range(len(point_xs)):
        temp_val = get_value_by_coordinates(img, dataset, gcs, pcs, extend, shape, [point_xs[i], point_ys[i]], coordinate_type='xy')
        vals.append(temp_val)
        if i == 0:
            dists.append(0)
        else:
            dists.append(dists[-1] + np.sqrt((point_xs[i] - point_xs[i - 1]) ** 2 + (point_ys[i] - point_ys[i - 1]) ** 2))
    return dists, vals

def ExtractDEMProfileOfAPolyline(X_list, Y_list, img, dataset, gcs, pcs, extend, shape):
    vals, dists = [], []
    for i in range(len(X_list)):
        temp_val = get_value_by_coordinates(img, dataset, gcs, pcs, extend, shape, [X_list[i], Y_list[i]], coordinate_type='xy')
        vals.append(temp_val)
        if i == 0:
            dists.append(0)
        else:
            dists.append(dists[-1] + np.sqrt((X_list[i] - X_list[i - 1]) ** 2 + (Y_list[i] - Y_list[i - 1]) ** 2))
    return dists, vals

def ExtractDEMProfileOfAPolylineWithEyot(LX, LY, LIX, LIY, RX, RY, RIX, RIY, CX, CY, img, dataset, gcs, pcs, extend, shape):
    vals, dists = [], []
    for i in range(len(CX)):
        if LIX[i] < -10000:
            temp_val = get_value_by_coordinates(img, dataset, gcs, pcs, extend, shape, [CX[i], CY[i]], coordinate_type='xy')
            vals.append(temp_val)   
        else:
            left_check_x, left_check_y, right_check_x, right_check_y = \
                (LX[i] + LIX[i]) / 2, (LY[i] + LIY[i]) / 2, (RX[i] + RIX[i]) / 2, (RY[i] + RIY[i]) / 2
            temp_val_1 = get_value_by_coordinates(img, dataset, gcs, pcs, extend, shape, [left_check_x, left_check_y], coordinate_type='xy')
            temp_val_2 = get_value_by_coordinates(img, dataset, gcs, pcs, extend, shape, [right_check_x, right_check_y], coordinate_type='xy')
            vals.append((temp_val_1 + temp_val_2) / 2)
        if i == 0:
            dists.append(0)
        else:
            dists.append(dists[-1] + np.sqrt((CX[i] - CX[i - 1]) ** 2 + (CY[i] - CY[i - 1]) ** 2))
    return dists, vals

def getDEMForTransects(LRX_list, RRX_list, LRY_list, RRY_list, img, dataset, gcs, pcs, extend, shape):
    D_Res, V_Res = [], []
    for ind in range(len(LRX_list)):
        x_1, x_2, y_1, y_2 = LRX_list[ind], RRX_list[ind], LRY_list[ind], RRY_list[ind]
        DISTS, VALS = ExtractDEMProfileOfATransect(x_1, x_2, y_1, y_2, img, dataset, gcs, pcs, extend, shape)
        D_Res.append(DISTS)
        V_Res.append(VALS)
    return D_Res, V_Res
    
#%% This cell designed for bank erosion
def getBankErosionInds(width_array, thresh):
    return list(np.where(width_array >= thresh)[0])

#%% This cell designed for getting usgs data.
def getUSGSData(URL, site_NO):
    str_1 = 'cb_'
    str_2 = '=on&format=html&site_no='
    str_3 = '&period=&begin_date='
    str_4 = '&end_date='
    head = {}
    head['User-Agent'] = 'Mozilla/5.0 (Linux; Android 4.1.1; Nexus 7 Build/JRO03D) AppleWebKit/535.19 (KHTML, like Gecko) Chrome/18.0.1025.166  Safari/535.19'
    
    code_list = []
    name_list = []
    start_list = []
    end_list = []
    Request_URL = URL + 'site_no=' + site_NO
    req = request.Request(Request_URL, headers=head)
    try:
        response = request.urlopen(req)
        html = response.read().decode('utf-8')
        soup_texts = BeautifulSoup(html, 'lxml') 
        table = soup_texts.find_all('div',class_="available_parameters_float1")
        table_soup = BeautifulSoup(str(table), 'lxml')
        for child in table_soup.table.tbody.children:
            if not child or child == '\n':
                continue
            else:
                temp = str(child)
            
                label = re.search('<label for=.{1,}</label>', temp)
                pattern = re.compile('[0-9]{4}-[0-9]{2}-[0-9]{2}')
                date = pattern.findall(temp)
                if label:
                    str_label = label.group(0)
                    label_cont = re.search('>[0-9]{5}.{1,}</font></label>', str_label)
                    if label_cont:
                        cont = label_cont.group(0)
                        cont_len = len(cont)
                        code_list.append(cont[1 : 6])
                        name_list.append(cont[7 : cont_len - 15])
                if date:
                    start_list.append(date[0])
                    if len(date) == 1:
                        end_list(datetime.datetime.now().strftime("%Y-%m-%d"))
                    else:
                        end_list.append(date[1])
                
    except error.URLError as e:
        if hasattr(e, 'code'):
            print("HTTPError")
            print(e.code)
        elif hasattr(e, 'reason'):
            print("URLError")
            print(e.reason)

    for j in range(len(name_list)):
        file = open(site_NO + '_' + code_list[j] + '.txt', 'w')
        date1 = datetime.datetime.strptime(start_list[j],"%Y-%m-%d")
        temp_day = date1
        date2 = datetime.datetime.strptime(end_list[j],"%Y-%m-%d")
        delta_day = (date2 - date1).days
        while delta_day > 60:
            temp_day = date1 + relativedelta(days = 60)
            delta_day = delta_day - 60
            new_URL = URL + str_1 + code_list[j] + str_2 + site_NO + str_3 + str(date1)[0:10] + str_4 + str(temp_day.strftime('%Y-%m-%d'))
            readhtmltable(new_URL, head, file, str(date1)[0:10], temp_day.strftime('%Y-%m-%d'))
            date1 = temp_day + relativedelta(days = 1)
            date_1 = date1.strftime('%Y-%m-%d')
            new_URL = URL + str_1 + code_list[j] + str_2 + site_NO + str_3 + str(date_1)[0:10] + str_4 + str(date2)[0:10]
            readhtmltable(new_URL, head, file, str(date_1)[0:10], str(date2)[0:10])
            file.close()
                

def readhtmltable(URL, head, file, start_date, end_date):
    date1 = datetime.datetime.strptime(start_date,"%Y-%m-%d")
    date2 = datetime.datetime.strptime(end_date,"%Y-%m-%d")
    delta_day = (date2 - date1).days
    count = 0
    print(URL)
    req = request.Request(URL, headers=head)
    try:
        response = request.urlopen(req)
        html = response.read().decode('utf-8')
        soup_texts = BeautifulSoup(html, 'lxml')
        tablehead = soup_texts.find_all('tbody')
        temp_t_r = 'unknown'
        for child in tablehead:
            tablecont = BeautifulSoup(str(child), 'lxml')
            record = tablecont.find_all('td')
            for r in record:
                t_r = re.search('[0-9]{2}\W[0-9]{2}.{1,}T', str(r))
                if t_r:
                    temp_t_r = t_r.group(0)
                    continue
                if temp_t_r == 'unknown':
                    continue
                
                count = count + 1
                delta_d = count - delta_day * (count // delta_day) - 1
                temp_date = date1 + relativedelta(days = delta_d)               
                v_r1 = re.search('[-+]?([0-9]+(\.[0-9]*)?|\.[0-9]+)', str(r))
                if v_r1:
                    file.write(str(temp_date)[0:10] + ',')
                    file.write(temp_t_r + ',')
                    file.write(v_r1.group(0))                   
                    file.write('\n')
        time.sleep(2)

    except error.URLError as e:
        if hasattr(e, 'code'):
            print("HTTPError")
            print(e.code)
        elif hasattr(e, 'reason'):
            print("URLError")
            print(e.reason)
            
def getTotalQ(gage_list, gage_signs):
    pass

#%% This cell designed for real-world distance and scaling computation.    
def getRealGlobalWorldDistance(lat_1, lon_1, lat_2, lon_2):
    R = 6373.0
    lon_1, lon_2, lat_1, lat_2 = math.radians(lon_1), math.radians(lon_2),\
                                 math.radians(lat_1), math.radians(lat_2)
    dlon = lon_2 - lon_1
    dlat = lat_2 - lat_1
    a = (math.sin(dlat / 2)) ** 2 + math.cos(lat_1) * math.cos(lat_2) * \
        (math.sin(dlon / 2)) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = R * c
    return distance

def getRealProjectedWorldDistance(lat_1, lon_1, lat_2, lon_2):
    pass


#%% This cell designed for delineation GUI

#%% This cell designed for reading georeference infos
def readImage(img_path):
    dataset = gdal.Open(img_path, GA_ReadOnly)
    geoTransform = dataset.GetGeoTransform()
    im_proj = dataset.GetProjection()
    return im_proj, geoTransform

#%% This cell designed for splitting result orthomap into pieces.

# extract np.array -> get transformation -> retransform
def SplitOrthoPhoto(ortho_map, save_dir, row_num, col_num):
    CheckIfFolderExist(save_dir)
    orig_row_size = ortho_map.RasterXSize
    orig_col_size = ortho_map.RasterYSize
    band = ortho_map.RasterCount

    for i in range(band):
        data = ortho_map.GetRasterBand(i+1).ReadAsArray()
        data = np.expand_dims(data , 2)
        if i == 0:
            allarrays = data
        else:
            allarrays = np.concatenate((allarrays, data), axis=2)
    data_dict = {'data':allarrays,'transform':ortho_map.GetGeoTransform(),\
                 'projection':ortho_map.GetProjection(),'bands':band,\
                     'width':orig_row_size,'height':orig_col_size}
    
    orig_data = data_dict.get('data')
    orig_transform = data_dict.get('transform')
    orig_size = orig_data.shape
    x = orig_transform[0]
    y = orig_transform[3]
    x_step = orig_transform[1]
    y_step = orig_transform[5]
    output_x_step = x_step
    output_y_step = y_step
    r_unit = orig_size[0]//row_num
    c_unit = orig_size[1]//col_num
    for r in range(row_num):
        for c in range(col_num):
            r_left = r_unit * r
            c_left = c_unit * c
            r_right = min((r + 1) * r_unit, orig_size[0])
            c_right = min((c + 1) * c_unit, orig_size[1])
            output_data = orig_data[r_left : r_right, c_left : c_right, :]
            output_transform = (x + c * output_x_step * r_unit,\
                output_x_step, 0, y + r * output_y_step * c_unit,\
                    0, output_y_step) 
            write_tif(f'{save_dir}/{r}_{c}.tif', output_data, output_transform, data_dict.get('projection'))


def write_tif(fn_out, im_data, transform,proj=None):
    if proj is None:
        proj = 'GEOGCS["WGS 84",\
                     DATUM["WGS_1984",\
                             SPHEROID["WGS 84",6378137,298.257223563, \
                                    AUTHORITY["EPSG","7030"]], \
                             AUTHORITY["EPSG","6326"]], \
                     PRIMEM["Greenwich",0, \
                            AUTHORITY["EPSG","8901"]], \
                     UNIT["degree",0.0174532925199433, \
                            AUTHORITY["EPSG","9122"]],\
                     AUTHORITY["EPSG","4326"]]'

    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32
    im_bands = min(im_data.shape)
    im_shape = list(im_data.shape)
    im_shape.remove(im_bands)
    im_height, im_width = im_shape
    band_idx = im_data.shape.index(im_bands)
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(fn_out, im_width, im_height, im_bands, datatype)

    # if dataset is not None:
    dataset.SetGeoTransform(transform)  
    dataset.SetProjection(proj)  

    if im_bands == 1:

        # print(im_data[:, 0,:].shape)
        if band_idx == 0:
            dataset.GetRasterBand(1).WriteArray(im_data[0, :, :])
        elif band_idx == 1:
            dataset.GetRasterBand(1).WriteArray(im_data[:, 0, :])
        elif band_idx == 2:
            dataset.GetRasterBand(1).WriteArray(im_data[:, :, 0])

    else:

        for i in range(im_bands):
            if band_idx == 0:
                dataset.GetRasterBand(i + 1).WriteArray(im_data[i, :, :])
            elif band_idx == 1:
                dataset.GetRasterBand(i + 1).WriteArray(im_data[:, i, :])
            elif band_idx == 2:
                dataset.GetRasterBand(i + 1).WriteArray(im_data[:, :, i])

    dataset.FlushCache()
    del dataset
    driver = None


def ExtractPolygonsFromOrthoPhoto(ortho_path, save_folder, save_file, x_coords, y_coords):
    coords = []
    coord_len = len(x_coords)
    for i in range(coord_len):
        coords.append((x_coords[i], y_coords[i]))
    poly = Polygon(coords)
    
    with rasterio.open(ortho_path) as src:
        out_image, out_transform = rasterio.mask.mask(src, [poly], crop=True)
        out_meta = src.meta

    out_meta.update(
        {
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform,
        }
    )
    CheckIfFolderExist(save_folder)
    with rasterio.open(save_folder + save_file, "w", **out_meta) as dest:
        dest.write(out_image)

def ReExtractWithIslandPolygon(ortho_path, save_folder, save_file, x_coords, y_coords,\
                               island_x, island_y):
    coords = []
    i_coords = []
    coord_len = len(x_coords)
    for i in range(coord_len):
        coords.append((x_coords[i], y_coords[i]))
    island_len = len(island_x)
    for j in range(island_len):
        i_coords.append((island_x[j], island_y[j]))
    poly = Polygon(coords, [i_coords])
    with rasterio.open(ortho_path) as src:
        out_image, out_transform = rasterio.mask.mask(src, [poly], crop=True)
        out_meta = src.meta

    out_meta.update(
        {
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform,
        }
    )
    CheckIfFolderExist(save_folder)
    with rasterio.open(save_folder + save_file, "w", **out_meta) as dest:
        dest.write(out_image)
    
def ConstructIslandPolygonCoords(island_left_x, island_left_y, island_right_x, island_right_y):
    return island_left_x + island_right_x[::-1], island_right_x + island_right_y[::-1]    

def ConstructTransectPolygon(left_x, left_y, mid_x, mid_y, right_x, right_y, ind):
    point_num = len(left_x)
    coords = []
    if ind == 0:
        left_l_x, left_r_x, right_l_x, right_r_x = left_x[ind], left_x[ind + 1], right_x[ind], right_x[ind + 1]
        left_l_y, left_r_y, right_l_y, right_r_y = left_y[ind], left_y[ind + 1], right_y[ind], right_y[ind + 1]
        left_m_x, right_m_x, left_m_y, right_m_y = (left_l_x + left_r_x) / 2, (right_l_x + right_r_x) / 2,\
                                                (left_l_y + left_r_y) / 2, (right_l_y + right_r_y) / 2        
        
    elif ind == point_num - 1:
        left_l_x, left_r_x, right_l_x, right_r_x = left_x[ind - 1], left_x[ind], right_x[ind - 1], right_x[ind]
        left_l_y, left_r_y, right_l_y, right_r_y = left_y[ind - 1], left_y[ind], right_y[ind - 1], right_y[ind]
        left_m_x, right_m_x, left_m_y, right_m_y = (left_l_x + left_r_x) / 2, (right_l_x + right_r_x) / 2,\
                                                (left_l_y + left_r_y) / 2, (right_l_y + right_r_y) / 2        
        
    else:
        left_l_x, left_r_x, right_l_x, right_r_x = (left_x[ind - 1] + left_x[ind]) / 2, (left_x[ind] + left_x[ind + 1]) / 2, \
                                                (right_x[ind - 1] + right_x[ind]) / 2, (right_x[ind] + right_x[ind + 1]) / 2
        left_l_y, left_r_y, right_l_y, right_r_y = (left_y[ind - 1] + left_y[ind]) / 2, (left_y[ind] + left_y[ind + 1]) / 2, \
                                                (right_y[ind - 1] + right_y[ind]) / 2, (right_y[ind] + right_y[ind + 1]) / 2        
        left_m_x, right_m_x, left_m_y, right_m_y = left_x[ind], right_x[ind], left_y[ind], right_y[ind]
    coords.append((left_l_x, left_l_y))
    coords.append((left_m_x, left_m_y))
    coords.append((left_r_x, left_r_y))
    coords.append((right_r_x, right_r_y))
    coords.append((right_m_x, right_m_y))
    coords.append((right_l_x, right_l_y))
    return Polygon(coords)

def ConstructTransectPolygonIfIslandExist(left_x, left_y, mid_x, mid_y, right_x, right_y, \
                        island_left_x, island_left_y, island_right_x, island_right_y, ind):
    orig_poly = ConstructTransectPolygon(left_x, left_y, mid_x, mid_y, right_x, right_y, ind)
    island_x, island_y = ConstructIslandPolygonCoords(island_left_x, island_left_y, island_right_x, island_right_y)
    island_coords = []
    island_len = len(island_x)
    for i in range(island_len):
        island_coords.append((island_x[i], island_y[i]))
    island_poly = Polygon(island_coords)
    return orig_poly.symmetric_difference(island_poly)

def ComputeAverageChannelSlope(dem_path, x_list, y_list, ind):
    img, dataset, gcs, pcs, extend, shape = get_tif_info(dem_path)
    selected_range = range(ind - 5, ind + 5)
    vals = []
    dist = []
    for i in selected_range:
        coordinates = [x_list[i], y_list[i]]
        temp_val = get_value_by_coordinates(img, dataset, gcs, pcs, extend, shape, coordinates, coordinate_type='xy')
        vals.append(temp_val)
        if i == 0:
            dist.append(0)
        else:
            dist.append(dist[i - 1] + \
                DistanceBetweenTwoPoints(x_list[i - 1], y_list[i - 1], x_list[i], y_list[i]))
    dist = sm.add_constant(dist)
    model = sm.OLS(vals, dist).fit()
    return model.params[1]

def ComputeSideSlopeForATransect(transect_dist_list, transect_elev_list, water_surface_elev, ind_1, ind_2, dist_3, stream_width, side_dist, choice = 0):
       
    ind_list, res_list = find_close_vals(transect_elev_list, water_surface_elev, thresh = 0.05)
    
    pure_ind_list = []
    for il in ind_list:
        if abs(transect_dist_list[il] - dist_3) <= stream_width / 2 + side_dist:
            pure_ind_list.append(il)
    if not pure_ind_list:
        left_ind = ind_1
        right_ind = ind_2
    else:
        left_ind = min(pure_ind_list)
        right_ind = max(pure_ind_list)
    left_elev, right_elev = transect_elev_list[left_ind], transect_elev_list[right_ind]
    left_dist, right_dist = transect_dist_list[left_ind], transect_dist_list[right_ind]
    
    left_side_ind, left_side_dist = find_nearest(transect_dist_list, left_dist - side_dist)
    right_side_ind, right_side_dist = find_nearest(transect_dist_list, right_dist + side_dist)
    if choice != 0:
        left_side_slope, right_side_slope = abs((left_side_dist - left_dist) / (abs(transect_elev_list[left_side_ind] - left_elev)+ 0.0001)),\
                                     abs((right_side_dist - right_dist) / (abs(transect_elev_list[right_side_ind] - right_elev) + 0.0001))
    else:
        left_xvals, left_yvals, right_xvals, right_yvals = transect_dist_list[left_side_ind : left_ind + 1], transect_elev_list[left_side_ind : left_ind + 1],\
                                                   transect_dist_list[right_ind : right_side_ind + 1][::-1], transect_elev_list[right_ind : right_side_ind + 1]
        left_xvals = sm.add_constant(left_xvals)
        left_model = sm.OLS(left_yvals, left_xvals).fit()
        right_xvals = sm.add_constant(right_xvals)
        right_model = sm.OLS(right_yvals, right_xvals).fit()    
        left_side_slope, right_side_slope = 1 / abs(left_model.params[1]), 1 / abs(right_model.params[1])                           
    return left_side_slope, right_side_slope

def ComputeSideSlopeForATransectIND(LFX, LFY, RFX, RFY, LRX, LRY, RRX, RRY, WIDTH, img, dataset, gcs, pcs, extend, shape, IND, choice = 0):
    left_bank_x, left_bank_y, right_bank_x, right_bank_y, left_ray_x, left_ray_y, right_ray_x, right_ray_y, stream_width = \
        LFX[IND], LFY[IND], RFX[IND], RFY[IND], LRX[IND], LRY[IND], RRX[IND], RRY[IND], WIDTH[IND]
    transect_dist_list, transect_elev_list = ExtractDEMProfileOfATransect(left_ray_x, \
                                    right_ray_x, left_ray_y, right_ray_y, img, dataset, gcs, pcs, extend, shape)
    ind_1, dist_1 =  find_nearest(transect_dist_list, np.sqrt((left_ray_x - left_bank_x) ** 2 + (left_ray_y - left_bank_y) ** 2))
    ind_2, dist_2 =  find_nearest(transect_dist_list, np.sqrt((left_ray_x - right_bank_x) ** 2 + (left_ray_y - right_bank_y) ** 2))
    ind_3, dist_3 =  find_nearest(transect_dist_list, np.sqrt((left_ray_x - (left_bank_x + right_bank_x) / 2) ** 2 + \
                                                              (left_ray_y - (left_bank_y + right_bank_y) / 2) ** 2 ))
    water_surface_elev = (transect_elev_list[ind_1] + transect_elev_list[ind_2] + transect_elev_list[ind_3]) / 3
    left_slopes, right_slopes = [], [] 
    max_side_dist = min(stream_width, 6)
    for side_dist in list(np.arange(0.5, max_side_dist, 0.5)):
        left_side_slope, right_side_slope = ComputeSideSlopeForATransect(transect_dist_list, transect_elev_list, water_surface_elev, \
                                                                     ind_1, ind_2, dist_3, stream_width, side_dist, choice = choice)
        left_slopes.append(left_side_slope)
        right_slopes.append(right_side_slope)
    return min(left_slopes), min(right_slopes)
    
def SaveXYFile(X_list, Y_list, file_folder, file_name):    
    CheckIfFolderExist(file_folder)
    with open(file_folder + file_name, 'w') as f:
        print('OBJECTID,ORIG_FID,X,Y\n', file = f)
        for i in range(len(X_list)):
            print('%d,%d,%f,%f\n'%(i + 1, 1, X_list[i], Y_list[i]), file = f)
    f.close()
    
#%% This cell deals with if a point locates within a polygon and compute the ray coordinates
def IsRayIntersectSegment(poi, s_poi, e_poi): #[x,y] [lng,lat]
    if s_poi[1] == e_poi[1]: 
        return False
    if s_poi[1] > poi[1] and e_poi[1] > poi[1]: 
        return False
    if s_poi[1] < poi[1] and e_poi[1] < poi[1]: 
        return False
    if s_poi[1] == poi[1] and e_poi[1] > poi[1]: 
        return False
    if e_poi[1] == poi[1] and s_poi[1] > poi[1]: 
        return False
    if s_poi[0] < poi[0] and e_poi[1] < poi[1]: 
        return False
    xseg = e_poi[0] - (e_poi[0] - s_poi[0]) * (e_poi[1] - poi[1]) / (e_poi[1] - s_poi[1]) 
    if xseg < poi[0]:
        return False
    return True  

def IsPointWithinPolygon(poi, poly):
    sinsc = 0 
    for epoly in poly: 
        for i in range(len(epoly) - 1): 
            s_poi = epoly[i]
            e_poi = epoly[i + 1]
            if IsRayIntersectSegment(poi, s_poi, e_poi):
                sinsc += 1 
    return True if sinsc % 2 == 1 else  False

#%% This cell designed for white-pixel detection

# scale --> pixel_num --> expand --> extract --> count_white
def getUnitLenAtALatitude(lat):
    earth_radius = 6371000
    earth_perimeter = 2 * np.pi * earth_radius
    lat_rad = lat * np.pi / 180
    lat_perimeter = earth_perimeter * np.cos(lat_rad)
    lat_unit = earth_perimeter / 360
    lon_unit = lat_perimeter / 360
    return lat_unit, lon_unit

def getWhiteCountRatio(input_img):
    return np.sum(input_img) / np.sum(np.where(input_img >= 250, 1, 0))

def ifWhitePixelSatisfied(input_img, thresh):
    return getWhiteCountRatio(input_img) >= thresh

def getScalingFactor(lat_unit, lon_unit):
    return np.sqrt(lat_unit ** 2 + lon_unit ** 2)

def getRealImgDelta(scaling, real_dist):
    return int(np.ceil(real_dist * scaling))

#%% This ceil designed for saving the indices into pckl or csv files
def contrustIndexMatrix(*args):
    arg_len = len(args)
    
    for i in range(arg_len):
        temp_array = np.array(args[i])
        if i == 0:
            res = temp_array
        else:
            res = np.vstack((res, temp_array))
    return res

def constructIndexVals(logical_array, array_list):
    basic_name = 'M'
    eval_str = 'contrustIndexMatrix('
    tab_array = []
    for i in range(len(logical_array)):
        if logical_array[i] == 1:
            exec('%s%d'%(basic_name, i+1) + '=array_list[%d]'%(i))
            eval_str += '%s%d,'%(basic_name, i+1)
            tab_array.append('%s%d'%(basic_name, i+1))
    eval_str += ')'
    res = eval(eval_str)
    return res, tab_array

def SavePdFile(res, tab_array, save_folder, save_file):
    tab_len = len(tab_array)
    for i in range(tab_len):
        temp_tab = tab_array[i]


#%% This cell designed for converting continuously data into categorized data.
def ConvertValsIntoClasses(vals, val_standards):
    val_array = np.array(vals)
    for j in range(len(val_standards)):        
        if j == 0:
            res = np.where(val_array >= val_standards[j], 1, 0)
        else:
            res += np.where(val_array >= val_standards[j], 1, 0)
    return res

#%% This cell designed for data input and organization
def ReadOneSideInput(csv_folder, csv_file):
    X_list = []
    Y_list = []
    count = 0
    with open(csv_folder + csv_file, 'r') as f:
        temp_lines = f.readlines()
        f.close()
    # print(csv_folder + csv_file)
    for l in temp_lines:
        
        if count == 0:
            count = count + 1
            continue
        else:
            
            line_split = l.split(',')
            if len(line_split) > 1:
                X_list.append(float(line_split[2]))
                Y_list.append(float(line_split[3]))
            else:
                line_split = l.split('\t')
                X_list.append(float(line_split[2]))
                Y_list.append(float(line_split[3]))
    return X_list, Y_list
   
def ReadTwoSideInputs(csv_folder, csv_file):
    LX_list, LY_list, RX_list, RY_list = [], [], [], []
    count = 0
    with open(csv_folder + csv_file, 'r') as f:
        temp_lines = f.readlines()
        f.close()
    for l in temp_lines:
        if count == 0:
            count = count + 1
            continue
        else:
            line_split = l.split(',')
            LX_list.append(float(line_split[0]))
            LY_list.append(float(line_split[1]))
            RX_list.append(float(line_split[2]))
            RY_list.append(float(line_split[3]))
    return LX_list, LY_list, RX_list, RY_list 

def ReadGTInputs(gt_folder, gt_file):
    DIST,DEPTH,WIDTH,BUFFER_L,BUFFER_R,EROSION_L,EROSION_R,VEG_COVER_L,VEG_COVER_R,WOOD, BANKTOP = [], [], [], [], [], [], [], [], [], [], []
    count = 0
    with open(gt_folder + gt_file, 'r') as f:
        temp_lines = f.readlines()
        f.close()
    for l in temp_lines:
        if count == 0:
            count = count + 1
            continue
        else:
            line_split = l.split(',')
            DIST.append(float(line_split[0]))
            DEPTH.append(float(line_split[1]))
            WIDTH.append(float(line_split[2]))
            BUFFER_L.append(float(line_split[3]))
            BUFFER_R.append(float(line_split[4]))
            EROSION_L.append(float(line_split[5]))
            EROSION_R.append(float(line_split[6]))
            VEG_COVER_L.append(float(line_split[7]))
            VEG_COVER_R.append(float(line_split[8]))
            WOOD.append(float(line_split[9]))
            BANKTOP.append(float(line_split[10]))
    return DIST,DEPTH,WIDTH,BUFFER_L,BUFFER_R,EROSION_L,EROSION_R,VEG_COVER_L,VEG_COVER_R,WOOD, BANKTOP

#%% This cell designed for organizing index values.
def getMType1(buffer_array, threshs):
    res = []
    for i in buffer_array:
        if i <= threshs[0]:
            res.append(1)
        elif i <= threshs[1]:
            res.append(2)
        elif i <= threshs[2]:
            res.append(3)
        else:
            res.append(4)
    return res

def getMType2(buffer_array, threshs):
    res = []
    for i in buffer_array:
        if i <= threshs[0]:
            res.append(4)
        elif i <= threshs[1]:
            res.append(3)
        elif i <= threshs[2]:
            res.append(2)
        else:
            res.append(1)
    return res

def getMType3(value, threshs):
    if value <= threshs[0]:
        res = 1
    elif value <= threshs[1]:
        res = 2
    elif value <= threshs[2]:
        res = 3
    else:
        res = 4
    return res

def getMType4(value, threshs):
    if value <= threshs[0]:
        res = 4
    elif value <= threshs[1]:
        res = 3
    elif value <= threshs[2]:
        res = 2
    else:
        res = 1
    return res

def getMType5(value, lower_threshs, higher_threshs):
    # e.g. if the thresh is 40% - 60%	30% - 40%, 60% - 70%	10% - 30%, 70% - 90%	< 10%,> 90%
    # Then input: [40, 30, 10] for lower_thresh, input: [60, 70, 90] for higher_thresh
    if value >= lower_threshs[0] and value <= higher_threshs[0]:
        res = 4
    elif value >= lower_threshs[1] and value <= higher_threshs[1]:
        res = 3
    elif value >= lower_threshs[2] and value <= higher_threshs[2]:
        res = 2
    else:
        res = 1
    return res

def getAverageMType1(vals, threshs):
    avg_val = np.mean(vals)
    return getMType3(avg_val, threshs)
    
def getAverageMType2(vals, threshs):
    avg_val = np.mean(vals)
    return getMType4(avg_val, threshs)

def getAverageValsForEachGTTransectRange(vals, GT_inds):
    res = []
    for gt_i in range(len(GT_inds)):
        start_ind = 0 if gt_i == 0 else int((GT_inds[gt_i] + GT_inds[gt_i - 1]) / 2)
        end_ind = len(vals) if gt_i == len(GT_inds) - 1 else int((GT_inds[gt_i] + GT_inds[gt_i + 1]) / 2)
        res.append(vals[start_ind:end_ind])
    return res

def computeFrequency(vals, unique_sorted_vals, upper = 4, lower = 1):
    res = []
    
    if len(unique_sorted_vals) < upper - lower + 1:
        unique_sorted_vals = list(range(lower, upper + 1))
    for usv in unique_sorted_vals:
        # print('aaaa')
        res.append(100 * len(np.where(np.array(vals) == usv)[0])/len(vals))
    # print(res)
    return res

def getUniqueSortedVals(vals):
    return list(set(vals))


#%% This cell designed for plot.

def PlotDoubleSideIndex(boundary_poly, row_num, col_num, left_points, \
                        right_points, left_vals, right_vals, val_standards, boundary_color,\
                        val_color_mat, save_path, line_thickness = 12):
# boundary_poly, left_points, right_points should be of the format: [[x1, y1], [x2, y2]]...
# boundary_color should be of the  format: [B,G,R]
# val_color_mat should be of the format: [[B1, G1, R1], [B2, G2, R2]...]
    left_point_num = len(left_points)
    right_point_num = len(right_points)
    empty_img = np.ones((row_num, col_num), np.uint8)
    empty_img *= 255
    left_con, right_con = ConvertValsIntoClasses(left_vals, val_standards), ConvertValsIntoClasses(right_vals, val_standards)
    
    cv2.fillPoly(empty_img, pts = [boundary_poly], color = boundary_color)
    for i in range(left_point_num):
        start_x = left_points[i][0] if i == 0 else (left_points[i][0] + left_points[i - 1][0]) / 2
        end_x = left_points[i][0] if i == left_point_num - 1 else (left_points[i][0] + left_points[i + 1][0]) / 2
        start_y = left_points[i][1] if i == 0 else (left_points[i][1] + left_points[i - 1][1]) / 2
        end_y = left_points[i][1] if i == left_point_num - 1 else (left_points[i][1] + left_points[i + 1][1]) / 2
        cv2.line(empty_img, (start_x, start_y), (end_x, end_y), val_color_mat[left_con[i]], line_thickness)
    
    for j in range(right_point_num):
        start_x = right_points[j][0] if j == 0 else (right_points[j][0] + right_points[j - 1][0]) / 2
        end_x = right_points[j][0] if j == right_point_num - 1 else (right_points[j][0] + right_points[j + 1][0]) / 2
        start_y = right_points[j][1] if j == 0 else (right_points[j][1] + right_points[j - 1][1]) / 2
        end_y = right_points[j][1] if j == right_point_num - 1 else (right_points[j][1] + right_points[j + 1][1]) / 2
        cv2.line(empty_img, (start_x, start_y), (end_x, end_y), val_color_mat[right_con[j]], line_thickness)
    cv2.imwrite(save_path, empty_img)

def plotXYLineWithTwoCategories(X, Y, Category):
    # Category 2 is the category we want to emphasize.
    plt.figure(figsize=(30,20))
    XC1, XC2, YC1, YC2 = [], [], [], []
    for i in range(len(X)):
        if Category[i] == 1:
            XC1.append(X[i])
            YC1.append(Y[i])
        else:
            XC2.append(X[i])
            YC2.append(Y[i])
    x = np.array(XC1)
    y = np.array(YC1)
    plt.scatter(x, y, color = '#fc8d59', s = 0.5)
    
    x = np.array(XC2)
    y = np.array(YC2)
    plt.scatter(x, y, color = '#91cf60', s = 5)
    
def plotLeftAndRightBank(left_X, left_Y, right_X, right_Y, mid_X, mid_Y):
    plt.figure(figsize=(20,10))
    x = np.array(left_X)
    y = np.array(left_Y)
    plt.scatter(x, y, color = 'hotpink', s = 0.5)
    
    x = np.array(right_X)
    y = np.array(right_Y)
    plt.scatter(x, y, color = '#88c999', s = 0.5)
    
    x = np.array(mid_X)
    y = np.array(mid_Y)
    plt.scatter(x, y, color = '#3288bd', s = 0.5)
    
    plt.show()
    plt.close()
    
def plotLeftAndRightBankWithEyot(left_X, left_Y, right_X, right_Y, mid_X, mid_Y, eyot_left_X, eyot_left_Y, \
                                 eyot_right_X, eyot_right_Y):
    plt.figure(figsize=(20,10))
    x = np.array(left_X)
    y = np.array(left_Y)
    plt.scatter(x, y, color = '#d73027', s = 0.5)
    
    x = np.array(right_X)
    y = np.array(right_Y)
    plt.scatter(x, y, color = '#fc8d59', s = 0.5)
    
    x = np.array(mid_X)
    y = np.array(mid_Y)
    plt.scatter(x, y, color = '#fee090', s = 0.5)
    
    x = np.array(eyot_left_X)
    y = np.array(eyot_left_Y)
    plt.scatter(x, y, color = '#91bfdb', s = 0.5)
    
    x = np.array(eyot_right_X)
    y = np.array(eyot_right_Y)
    plt.scatter(x, y, color = '#4575b4', s = 0.5)
    
    plt.show()
    plt.close()

def plotBoundMidLineBendAndPools(boundary_x, boundary_y, mid_x, mid_y, pool_inds, bend_inds, window_size = (32, 8)):
    fig = plt.figure(figsize=(30, 15))
    ax = fig.add_subplot(1,1,1)
    ax.fill(boundary_x, boundary_y, "g", alpha = .1) 
    for i in range(len(mid_x) - 1):
        if i in pool_inds:
            ax.plot([mid_x[i], mid_x[i + 1]], [mid_y[i], mid_y[i + 1]], color = '#1f78b4', linewidth = 3.0)
        else:
            ax.plot([mid_x[i], mid_x[i + 1]], [mid_y[i], mid_y[i + 1]], color = '#a6cee3', linewidth = 3.0)
    for k in bend_inds:
        ax.plot(mid_x[k], mid_y[k], linestyle='', marker='^', markeredgecolor= '#ff7f00', \
                markerfacecolor = '#ff7f00', markersize = 10)

def plotLeftAndRightBankWithRays(left_X, left_Y, right_X, right_Y, mid_X, mid_Y, Left_X_Ray, Left_Y_Ray, \
                                 Right_X_Ray, Right_Y_Ray, intervals):
    plt.figure(figsize=(20,10))
    x = np.array(left_X)
    y = np.array(left_Y)
    plt.scatter(x, y, color = '#d73027', s = 0.5)
    
    x = np.array(right_X)
    y = np.array(right_Y)
    plt.scatter(x, y, color = '#fc8d59', s = 0.5)
    
    x = np.array(mid_X)
    y = np.array(mid_Y)
    plt.scatter(x, y, color = '#fee090', s = 0.5)
    
    
    Ray_num = len(Left_X_Ray)
    
    for i in range(0, Ray_num, intervals):
        plt.plot([Left_X_Ray[i], Right_X_Ray[i]], [Left_Y_Ray[i], Right_Y_Ray[i]], color = '#4575b4', linewidth = 2)
    
    plt.show()
    plt.close()
    
def plotLeftAndRightBankWithGTRays(left_X, left_Y, right_X, right_Y, mid_X, mid_Y, Left_X_Ray, Left_Y_Ray, \
                                 Right_X_Ray, Right_Y_Ray, gt_inds):
    plt.figure(figsize=(20,10))
    x = np.array(left_X)
    y = np.array(left_Y)
    plt.scatter(x, y, color = '#d73027', s = 0.5)
    
    x = np.array(right_X)
    y = np.array(right_Y)
    plt.scatter(x, y, color = '#fc8d59', s = 0.5)
    
    x = np.array(mid_X)
    y = np.array(mid_Y)
    plt.scatter(x, y, color = '#fee090', s = 0.5)
    
    for i in gt_inds:
        plt.plot([Left_X_Ray[i], Right_X_Ray[i]], [Left_Y_Ray[i], Right_Y_Ray[i]], color = '#4575b4', linewidth = 2)
    
    plt.show()
    plt.close()
    
def plotParaMeterValues(x_vals, y_vals, x_label, y_label, x_lim = [], y_lim = [], window_size = (32, 8), font_size = 30):

    fig=plt.figure(figsize=(window_size[0], window_size[1]))
    ax=fig.add_subplot(1,1,1)
    if not x_lim:
        x_min, x_max = min(x_vals), max(x_vals) + (max(x_vals) - min(x_vals)) * 0.01
    else:
        x_min, x_max = x_lim[0], x_lim[1]

    if not y_lim:
        y_min, y_max = min(y_vals), max(y_vals) + (max(y_vals) - min(y_vals)) * 0.01
    else:
        y_min, y_max = y_lim[0], y_lim[1]
    C = '#252525'
    ax.plot(x_vals, y_vals, color = C, linewidth = 3.0)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.tick_params(axis='both', which='major', labelsize=font_size)
    #ax.fill_between(iters, r1, r2, color=color, alpha=0.2)
    ax.set_xlabel(x_label, fontsize=font_size)
    ax.set_ylabel(y_label, fontsize=font_size)
    
def plotParaMeterValuesWithBar(x_vals, y_vals, bar_val, x_label, y_label, x_lim = [], y_lim = [], window_size = (32, 8), font_size = 30):

    fig=plt.figure(figsize=(window_size[0], window_size[1]))
    ax=fig.add_subplot(1,1,1)
    if not x_lim:
        x_min, x_max = min(x_vals), max(x_vals) + (max(x_vals) - min(x_vals)) * 0.01
    else:
        x_min, x_max = x_lim[0], x_lim[1]

    if not y_lim:
        y_min, y_max = min(y_vals), max(y_vals) + (max(y_vals) - min(y_vals)) * 0.01
    else:
        y_min, y_max = y_lim[0], y_lim[1]
    C = '#252525'
    ax.plot(x_vals, y_vals, color = C, linewidth = 3.0)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.tick_params(axis='both', which='major', labelsize=font_size)
    #ax.fill_between(iters, r1, r2, color=color, alpha=0.2)
    ax.set_xlabel(x_label, fontsize=font_size)
    ax.set_ylabel(y_label, fontsize=font_size)
    ax.plot([x_min, x_max], [bar_val, bar_val])

def plotTwoParaMeterValues(x_vals1, y_vals1, x_vals2, y_vals2, x_label, y_label, x_lim, y_lim, \
                           l_legend = 'Left', r_legend = 'Right', window_size = (32, 8), font_size = 30):

    fig=plt.figure(figsize=(window_size[0], window_size[1]))
    ax=fig.add_subplot(1,1,1)
    C1 = '#252525'
    C2 = '#969696'
    ax.plot(x_vals1, y_vals1, color = C1, linewidth = 3.0)
    ax.plot(x_vals2, y_vals2, color = C2, linewidth = 3.0, linestyle = '--')
    ax.legend([l_legend, r_legend], loc = 'upper left', fontsize = font_size)
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.tick_params(axis='both', which='major', labelsize=font_size)
    #ax.fill_between(iters, r1, r2, color=color, alpha=0.2)
    ax.set_xlabel(x_label, fontsize=font_size)
    ax.set_ylabel(y_label, fontsize=font_size)

def plotParaMeterValuesWithPoints(x_vals, y_vals, x_points, y_points, x_label, y_label, window_size = (32, 8)):
    
    fig=plt.figure(figsize=(window_size[0], window_size[1]))
    ax=fig.add_subplot(1,1,1)
    
    C = '#252525'
    ax.plot(x_vals, y_vals, color = C, linewidth = 3.0)
    
    for i in range(len(x_points)):
        pass
    ax.tick_params(axis='both', which='major', labelsize=28)
    #ax.fill_between(iters, r1, r2, color=color, alpha=0.2)
    ax.set_xlabel(x_label, fontsize=28)
    ax.set_ylabel(y_label, fontsize=28)

def PlotParaMeterValsWithGT(x_vals, y_vals, GT_x, GT_y, x_label, y_label, x_lim = [], y_lim = [], window_size = (32, 8), font_size = 30):

    fig=plt.figure(figsize=(window_size[0], window_size[1]))
    ax=fig.add_subplot(1,1,1)
    
    if not x_lim:
        x_min, x_max = min(x_vals), max(x_vals) + (max(x_vals) - min(x_vals)) * 0.01
    else:
        x_min, x_max = x_lim[0], x_lim[1]

    if not y_lim:
        y_min, y_max = min(y_vals), max(y_vals) + (max(y_vals) - min(y_vals)) * 0.01
    else:
        y_min, y_max = y_lim[0], y_lim[1]
        
    C, GT_C = '#252525', '#fdb462'
    
    ax.plot(x_vals, y_vals, color = C, linewidth = 3.0)
    
    ax.plot(GT_x, GT_y, linestyle='', marker='o', markeredgecolor= GT_C, markerfacecolor = GT_C, markersize = 15)
    
    
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.tick_params(axis='both', which='major', labelsize=font_size)
    #ax.fill_between(iters, r1, r2, color=color, alpha=0.2)
    ax.set_xlabel(x_label, fontsize=font_size)
    ax.set_ylabel(y_label, fontsize=font_size)

def PlotTwoParaMeterValsWithGT(x_vals1, y_vals1, x_vals2, y_vals2, GT_x1, GT_y1, GT_x2, GT_y2, x_label, y_label,\
                               l_legend = 'Left', r_legend = 'Right', x_lim = [], y_lim = [], window_size = (32, 8), font_size = 30):

    fig=plt.figure(figsize=(window_size[0], window_size[1]))
    ax=fig.add_subplot(1,1,1)
    
    C1, C2, GT_C1, GT_C2 = '#252525', '#969696', '#fdb462', '#6a3d9a'
    

    ax.plot(x_vals1, y_vals1, color = C1, linewidth = 3.0)
    ax.plot(x_vals2, y_vals2, color = C2, linewidth = 3.0, linestyle = '--')
    
    ax.plot(GT_x1, GT_y1, linestyle='', marker='o', markeredgecolor= GT_C1, markerfacecolor = GT_C1, markersize = 15)
    ax.plot(GT_x2, GT_y2, linestyle='', marker='^', markeredgecolor= GT_C2, markerfacecolor = GT_C2, markersize = 15)
    
    ax.legend([l_legend, r_legend, l_legend + '_GT', r_legend + '_GT'], loc = 'upper left', fontsize = 20)
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.tick_params(axis='both', which='major', labelsize=font_size)
    #ax.fill_between(iters, r1, r2, color=color, alpha=0.2)
    ax.set_xlabel(x_label, fontsize=font_size)
    ax.set_ylabel(y_label, fontsize=font_size)

def plotOneSideMetric(mid_x, mid_y, mid_vals, boundary_x,\
                       boundary_y, Line_Colors = ['#d7191c', '#fdae61', '#a6d96a', '#1a9641'],\
                      FillColor = "#92c5de", FillAlpha = 0.1, Line_Width = 3, is_tick = False, window_size = (30, 15)):
    fig = plt.figure(figsize=(window_size[0], window_size[1]))
    ax = fig.add_subplot(1,1,1)
    ax.fill(boundary_x, boundary_y, FillColor, alpha = .1) 
    ax.tick_params(left = is_tick, right = is_tick , labelleft = is_tick ,
                    labelbottom = is_tick, bottom = is_tick)
    for i in range(len(mid_x) - 1):
        if mid_vals[i] == mid_vals[i + 1]:
            ax.plot([mid_x[i], mid_x[i + 1]], [mid_y[i], mid_y[i + 1]], \
                    color = Line_Colors[mid_vals[i] - 1], linewidth = Line_Width)
        else:
            ax.plot([mid_x[i], (mid_x[i] + mid_x[i + 1]) / 2], [mid_y[i], \
                        (mid_y[i] + mid_y[i + 1]) / 2], color = Line_Colors[mid_vals[i] - 1], \
                        linewidth = Line_Width)
            ax.plot([(mid_x[i] + mid_x[i + 1]) / 2, mid_x[i + 1]], [\
                        (mid_y[i] + mid_y[i + 1]) / 2, mid_y[i + 1]], \
                        color = Line_Colors[mid_vals[i + 1] - 1], linewidth = Line_Width)

def PlotParaMeterValsWithTwoGTs(x_vals1, y_vals1, GT_x1, GT_y1, GT_x2, GT_y2, x_label, y_label,\
                               legend_1 = 'Reference', legend_2 = 'Validation', x_lim = [], y_lim = [], window_size = (32, 8), font_size = 30):

    fig=plt.figure(figsize=(window_size[0], window_size[1]))
    ax=fig.add_subplot(1,1,1)
    
    C1, GT_C1, GT_C2 = '#252525','#fdb462', '#6a3d9a'
    

    ax.plot(x_vals1, y_vals1, color = C1, linewidth = 3.0)
    
    ax.plot(GT_x1, GT_y1, linestyle='', marker='o', markeredgecolor= GT_C1, markerfacecolor = GT_C1, markersize = 15)
    ax.plot(GT_x2, GT_y2, linestyle='', marker='^', markeredgecolor= GT_C2, markerfacecolor = GT_C2, markersize = 15)
    
    
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.tick_params(axis='both', which='major', labelsize=font_size)
    #ax.fill_between(iters, r1, r2, color=color, alpha=0.2)
    ax.set_xlabel(x_label, fontsize=font_size)
    ax.set_ylabel(y_label, fontsize=font_size)

def plotTwoSideMetrics(left_bank_x, left_bank_y, right_bank_x, right_bank_y, left_vals, right_vals, boundary_x,\
                       boundary_y, Line_Colors = ['#d7191c', '#fdae61', '#a6d96a', '#1a9641'],\
                      FillColor = "#92c5de", FillAlpha = 0.1, Line_Width = 3, is_tick = False, window_size = (30, 15)):
    fig = plt.figure(figsize=(window_size[0], window_size[1]))
    ax = fig.add_subplot(1,1,1)
    ax.fill(boundary_x, boundary_y, FillColor, alpha = .1) 
    ax.tick_params(left = is_tick, right = is_tick , labelleft = is_tick ,
                    labelbottom = is_tick, bottom = is_tick)
    for i in range(len(left_bank_x) - 1):
        if left_vals[i] == left_vals[i + 1]:
            ax.plot([left_bank_x[i], left_bank_x[i + 1]], [left_bank_y[i], left_bank_y[i + 1]], \
                    color = Line_Colors[left_vals[i] - 1], linewidth = Line_Width)
        else:
            ax.plot([left_bank_x[i], (left_bank_x[i] + left_bank_x[i + 1]) / 2], [left_bank_y[i], \
                        (left_bank_y[i] + left_bank_y[i + 1]) / 2], color = Line_Colors[left_vals[i] - 1], \
                        linewidth = Line_Width)
            ax.plot([(left_bank_x[i] + left_bank_x[i + 1]) / 2, left_bank_x[i + 1]], [\
                        (left_bank_y[i] + left_bank_y[i + 1]) / 2, left_bank_y[i + 1]], \
                        color = Line_Colors[left_vals[i + 1] - 1], linewidth = Line_Width)
    for j in range(len(right_bank_x) - 1):
        if right_vals[j] == right_vals[j + 1]:
            ax.plot([right_bank_x[j], right_bank_x[j + 1]], [right_bank_y[j], right_bank_y[j + 1]], \
                    color = Line_Colors[right_vals[j] - 1], linewidth = Line_Width)
        else:
            ax.plot([right_bank_x[j], (right_bank_x[j] + right_bank_x[j + 1]) / 2], [right_bank_y[j], \
                        (right_bank_y[j] + right_bank_y[j + 1]) / 2], color = Line_Colors[right_vals[j] - 1], \
                        linewidth = Line_Width)
            ax.plot([(right_bank_x[j] + right_bank_x[j + 1]) / 2, right_bank_x[j + 1]], [\
                        (right_bank_y[j] + right_bank_y[j + 1]) / 2, right_bank_y[j + 1]], \
                        color = Line_Colors[right_vals[j + 1] - 1], linewidth = Line_Width)

def plotOneSideMetricWithGT(mid_x, mid_y, mid_vals, GT_val, GT_inds, boundary_x,\
                       boundary_y, Line_Colors = ['#d7191c', '#fdae61', '#a6d96a', '#1a9641'],\
                      FillColor = "#92c5de", FillAlpha = 0.1, Line_Width = 3, is_tick = False, window_size = (30, 15)):
    fig = plt.figure(figsize=(window_size[0], window_size[1]))
    ax = fig.add_subplot(1,1,1)
    ax.fill(boundary_x, boundary_y, FillColor, alpha = .1) 
    ax.tick_params(left = is_tick, right = is_tick , labelleft = is_tick ,
                    labelbottom = is_tick, bottom = is_tick)
    for i in range(len(mid_x) - 1):
        if mid_vals[i] == mid_vals[i + 1]:
            ax.plot([mid_x[i], mid_x[i + 1]], [mid_y[i], mid_y[i + 1]], \
                    color = Line_Colors[mid_vals[i] - 1], linewidth = Line_Width)
        else:
            ax.plot([mid_x[i], (mid_x[i] + mid_x[i + 1]) / 2], [mid_y[i], \
                        (mid_y[i] + mid_y[i + 1]) / 2], color = Line_Colors[mid_vals[i] - 1], \
                        linewidth = Line_Width)
            ax.plot([(mid_x[i] + mid_x[i + 1]) / 2, mid_x[i + 1]], [\
                        (mid_y[i] + mid_y[i + 1]) / 2, mid_y[i + 1]], \
                        color = Line_Colors[mid_vals[i + 1] - 1], linewidth = Line_Width)
    for gt_i in GT_inds:
        gt_x, gt_y = mid_x[gt_i], mid_y[gt_i]
        GT_cat = mid_vals[gt_i]
        GT_C = Line_Colors[GT_cat - 1]
        ax.plot(gt_x, gt_y, linestyle='', marker='o', markeredgecolor= GT_C, markerfacecolor = GT_C, markersize = 10)


def plotTwoSideMetricsWithGT(left_bank_x, left_bank_y, right_bank_x, right_bank_y, left_vals, right_vals, GT_left, GT_right, GT_inds,\
                       boundary_x, boundary_y, Line_Colors = ['#d7191c', '#fdae61', '#a6d96a', '#1a9641'],\
                      FillColor = "#92c5de", FillAlpha = 0.1, Line_Width = 3, is_tick = False, window_size = [30, 15]):
    fig = plt.figure(figsize=(window_size[0], window_size[1]))
    ax = fig.add_subplot(1,1,1)
    ax.fill(boundary_x, boundary_y, FillColor, alpha = .1) 
    ax.tick_params(left = is_tick, right = is_tick , labelleft = is_tick ,
                    labelbottom = is_tick, bottom = is_tick)
    for i in range(len(left_bank_x) - 1):
        if left_vals[i] == left_vals[i + 1]:
            ax.plot([left_bank_x[i], left_bank_x[i + 1]], [left_bank_y[i], left_bank_y[i + 1]], \
                    color = Line_Colors[left_vals[i] - 1], linewidth = Line_Width)
        else:
            ax.plot([left_bank_x[i], (left_bank_x[i] + left_bank_x[i + 1]) / 2], [left_bank_y[i], \
                        (left_bank_y[i] + left_bank_y[i + 1]) / 2], color = Line_Colors[left_vals[i] - 1], \
                        linewidth = Line_Width)
            ax.plot([(left_bank_x[i] + left_bank_x[i + 1]) / 2, left_bank_x[i + 1]], [\
                        (left_bank_y[i] + left_bank_y[i + 1]) / 2, left_bank_y[i + 1]], \
                        color = Line_Colors[left_vals[i + 1] - 1], linewidth = Line_Width)
    for j in range(len(right_bank_x) - 1):
        if right_vals[j] == right_vals[j + 1]:
            ax.plot([right_bank_x[j], right_bank_x[j + 1]], [right_bank_y[j], right_bank_y[j + 1]], \
                    color = Line_Colors[right_vals[j] - 1], linewidth = Line_Width)
        else:
            ax.plot([right_bank_x[j], (right_bank_x[j] + right_bank_x[j + 1]) / 2], [right_bank_y[j], \
                        (right_bank_y[j] + right_bank_y[j + 1]) / 2], color = Line_Colors[right_vals[j] - 1], \
                        linewidth = Line_Width)
            ax.plot([(right_bank_x[j] + right_bank_x[j + 1]) / 2, right_bank_x[j + 1]], [\
                        (right_bank_y[j] + right_bank_y[j + 1]) / 2, right_bank_y[j + 1]], \
                        color = Line_Colors[right_vals[j + 1] - 1], linewidth = Line_Width)
    for gt_i in GT_inds:
        left_gt_x, left_gt_y, right_gt_x, right_gt_y = left_bank_x[gt_i], left_bank_y[gt_i], right_bank_x[gt_i], right_bank_y[gt_i]
        left_cat, right_cat = left_vals[gt_i], right_vals[gt_i]
        GT_C1, GT_C2 = Line_Colors[left_cat - 1], Line_Colors[right_cat - 1]
        ax.plot(left_gt_x, left_gt_y, linestyle='', marker='o', markeredgecolor= GT_C1, markerfacecolor = GT_C1, markersize = 10)
        ax.plot(right_gt_x, right_gt_y, linestyle='', marker='o', markeredgecolor= GT_C2, markerfacecolor = GT_C2, markersize = 10)
 
def plotHist(freqs, labels, Hist_Colors = ['#d7191c', '#fdae61', '#a6d96a', '#1a9641'], window_size = [8, 4], is_labeled = False):
    TNR = {'fontname':'Times New Roman'}
    plt.figure(figsize = window_size)
    figure,axes = plt.subplots(1, 1, figsize = window_size, dpi = 1000)
    plt.bar(range(len(freqs)), freqs, tick_label = labels, color = Hist_Colors)
    x = np.arange(len(freqs))
    if is_labeled:
        for i, j in zip(x, freqs):
            plt.text(i, j + 0.05, '%.2f%%' % j, ha = 'center', va = 'bottom', fontsize = 18, **TNR) 
    [axes.spines[loc_axis].set_visible(False) for loc_axis in ['top','right','left']] 
    axes.set_yticks([]) 
    axes.set_xticks([])
           
 
def plotTwoBarHists(x1, x2, y_labels, edge_color = '#252525', Line_Colors = ['#d7191c', '#fdae61', '#a6d96a', '#1a9641'], scale = 1.2, window_size = (12, 8), font_size = 20):
    font = {'family': 'Times New Roman',
            'size': font_size,
            }
    sns.set(font_scale = scale)
    
    plt.rc('font', family = 'Times New Roman')
    fig = plt.figure(figsize = window_size)
    ax = fig.add_subplot(1,1,1)
    ax.invert_yaxis()
    ax.grid(False)

    
    # We need to create the color mat
    x1_color, x2_color = [], []
    for i in x1:
        x1_color.append(Line_Colors[i - 1])
    for j in x2:
        x2_color.append(Line_Colors[j - 1])
    plt.barh(range(len(x1)), [- x1[xind] for xind in range(len(x1))], color = x1_color, edgecolor = edge_color)
    plt.barh(range(len(x2)), x2, color = x2_color, edgecolor = 'black')
    
    #plt.plot([0, 0], [-0.5, 3.5], color = 'black')
    
    x_max = max(max(x1), max(x2))
    
    plt.xlim((- x_max - 0.5, x_max + 0.5))
    
    x_tick_locs = list(range(0, - x_max - 1, -1)) + list(range(1,  x_max + 1, 1))
    x_tick_labels = [str(abs(xt)) for xt in x_tick_locs]
    y_tick_locs = list(range(0, len(y_labels)))
    
    ax.patch.set_facecolor('white')
    ax.patch.set_edgecolor('black')  

    ax.patch.set_linewidth(1.5)  
    
    plt.xticks(x_tick_locs, x_tick_labels, font = font)
    plt.yticks(y_tick_locs, y_labels, font = font)

def plotOneBarHist(x1,y_labels, edge_color = '#252525', Line_Colors = ['#d7191c', '#fdae61', '#a6d96a', '#1a9641'], scale = 1.2, window_size = (8, 12), font_size = 20):
    font = {'family': 'Times New Roman',
            'size': font_size,
            }
    sns.set(font_scale = scale)
    
    plt.rc('font', family = 'Times New Roman')
    fig = plt.figure(figsize = window_size)
    ax = fig.add_subplot(1,1,1)
    ax.invert_yaxis()
    ax.grid(False)

    
    # We need to create the color mat
    x1_color = []
    for i in x1:
        x1_color.append(Line_Colors[i - 1])
    
    plt.barh(range(len(x1)), x1, color = x1_color, edgecolor = edge_color)

    #plt.plot([0, 0], [-0.5, 3.5], color = 'black')
    
    x_max = max(x1)
    
    plt.xlim((- 0.5, x_max + 0.5))
    
    x_tick_locs = list(range(0,  x_max + 1, 1))
    x_tick_labels = [str(abs(xt)) for xt in x_tick_locs]
    y_tick_locs = list(range(0, len(y_labels)))
    
    ax.patch.set_facecolor('white')
    ax.patch.set_edgecolor('black')  

    ax.patch.set_linewidth(1.5)  
    
    plt.xticks(x_tick_locs, x_tick_labels, font = font)
    plt.yticks(y_tick_locs, y_labels, font = font)