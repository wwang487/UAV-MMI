# -*- coding: utf-8 -*-
"""
Created on Sat Oct  1 02:29:47 2022

@author: wei wang
"""

import math

def ComputeAFOV(camera_diameter, focus):
    # input unit should be mm
    camera_diameter = camera_diameter / 1000
    focus = focus / 1000
    angle = math.atan(camera_diameter / (2 * focus))
    return 2 * 180 * angle / math.pi

def ComputeHorizontalDist(AFOV, flight_elev):
    return 2 * flight_elev * math.tan(AFOV/2)

def ComputeFlightInterval1(hori_distance, overlap_ratio):
    return hori_distance * (1 - overlap_ratio) / (1 + overlap_ratio)

def ComputeFlightInterval2(hori_distance, overlap_ratio):
    return hori_distance * (1 - overlap_ratio)
