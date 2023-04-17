# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 02:27:48 2022

@author: bbean
"""
import cv2
import numpy as np
import math
import copy
import FloydSmoothing
import d_star_lite_optimized
import matplotlib.pyplot as plt

def StreamOrientationComputation(stream_img):
    # Herein, the shape image should be 255 (or any valid positive values) 
    """
    This function extracts the orientation and contours of a stream bw img with buffer.
    """
    if len(stream_img.shape) == 3:
        stream_img = cv2.cvtColor(stream_img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(stream_img, 0, 255, 0)
    cntrs = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cntrs = cntrs[0] if len(cntrs) == 2 else cntrs[1]
    rotrect = cv2.minAreaRect(cntrs[0])
    # box = cv2.boxPoints(rotrect)
    # box = np.int0(box)
    angle = rotrect[-1]
    # print(angle)
    # if angle < -45:
    #     angle = -(90 + angle)
    # else:
    #     angle = angle
    return angle, cntrs

def StreamMeanComputation(stream_img):
    """
    This function returns the mid point of stream area
    """
    if len(stream_img.shape) == 3:
        stream_img = cv2.cvtColor(stream_img, cv2.COLOR_BGR2GRAY)
    coords = np.where(stream_img > 0)
    
    return (np.mean(coords[1]), np.mean(coords[0]))

def getABCofAngle(angle, mid_point):
    """
    This function returns A, B, C coefs of a line passes a point with an angle.
    """
    # y = kx + b, b = y0 - k x0, kx - y + b = 0
    # x = x0, x - x0 = 0
    angle_rad = math.pi * angle / 180 
    if abs(angle_rad - math.pi / 2) < 0.000001:
        A, B, C = 1, 0, - mid_point[0]
    else:
        k = math.atan(angle_rad)
        b = mid_point[1] - k * mid_point[0]
        A, B, C = k, -1, b
    return A, B, C

def computeABCofASegment(x1, y1, x2, y2):
    """
    This function returns A, B, C coefs of the segment connecting point 1 and 2.
    """
    if x1 == x2:
        return 1, 0, - x1
    elif y1 == y2:
        return 0, 1, -y1
    else:
        k = (y2 - y1)/(x2 - x1)
        b = y1 - k * x1
        return k, -1, b

def get_distance_from_point_to_line(point, A, B, C):
    """
    This function returns the distance from a point to a line with A, B, C coefs
    """
    distance = np.abs(A * point[0] + B * point[1] + C) / (np.sqrt(A**2 + B**2))
    return distance

def get_distance_between_two_points(x1, y1, x2, y2):
    """
    This function returns the distance between two points
    """
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def getRefLineStartAndEndPoint(A, B, C, row_num, col_num):
    """
    This function returns the endpoints of a reference line with A, B, C coefs,
        reference line is a line that determines if a point is in the top/bottom
        side of this line
    """
    if B == 0:
        x1, y1, x2, y2 = - C / A, 0, - C / A, row_num - 1
    elif A == 0:
        x1, y1, x2, y2 = 0, - C / B, col_num - 1, - C / B
    else:
        x1, y1, x2, y2 = - C / A, 0, (- B * (row_num - 1) - C) / A, row_num - 1
        
    if x1 > x2:
        return x2, y2, x1, y1
    else:
        return x1, y1, x2, y2

def getSideofAPointOverALine(x1, y1, x2, y2, xp, yp):
    """
    This function returns the side of (xp, yp) related to the segment connecting point 1
    and point 2. 1 means above, 0 means on, -1 means below.
    """
    # 1 and 2 and line points, p are the points to check
    A, B, C = y2 - y1, x1 - x2, x2 * y1 - x1 * y2
    D = A * xp + B * yp + C
    if D > 0:
        return 1
    elif D == 0:
        return 0
    else:
        return -1

def getStreamIntervalRange(stream_img):
    """
    This function returns the range of bw region in a stream img
    """
    angle, cntrs = StreamOrientationComputation(stream_img)
    mid_point = StreamMeanComputation(stream_img)
    row_num, col_num = stream_img.shape[:2]
    cnt = cntrs[0]
    cnt_num = cnt.shape[0]
    max_positive_dist, max_negative_dist = 0, 0
    A, B, C = getABCofAngle(angle, mid_point)
    xr_1, yr_1, xr_2, yr_2 = getRefLineStartAndEndPoint(A, B, C, row_num, col_num)
    for i in range(cnt_num):
        point_x, point_y = cnt[i, 0, 0], cnt[i, 0, 1]
        dist = get_distance_from_point_to_line([point_x, point_y], A, B, C)
        direction = getSideofAPointOverALine(xr_1, yr_1, xr_2, yr_2, point_x, point_y)
        if direction == -1:
            if dist > max_positive_dist:
                max_positive_dist = dist
        else:
            if dist > max_negative_dist:
                max_negative_dist = dist
    return max_positive_dist, max_negative_dist

def getBoundaryIntervalRange(stream_img, boundary):
    # boundary should be a 2-D array, first column should be x, and second column should be y.
    angle, cntrs = StreamOrientationComputation(stream_img)
    mid_point = StreamMeanComputation(stream_img)
    row_num, col_num = stream_img.shape[:2]
    boundary_num = boundary.shape[0]
    max_positive_dist, max_negative_dist = 0, 0
    A, B, C = getABCofAngle(angle, mid_point)
    xr_1, yr_1, xr_2, yr_2 = getRefLineStartAndEndPoint(A, B, C, row_num, col_num)
    for i in range(boundary_num):
        temp_boundary = boundary[i]
        point_x, point_y = temp_boundary[0], temp_boundary[1]
        dist = get_distance_from_point_to_line([point_x, point_y], A, B, C)
        # print(dist)
        direction = getSideofAPointOverALine(xr_1, yr_1, xr_2, yr_2, point_x, point_y)
        if direction == -1:
            if dist > max_positive_dist:
                max_positive_dist = dist
        else:
            if dist > max_negative_dist:
                max_negative_dist = dist
    return max_positive_dist, max_negative_dist

def getParallelABCList(midpoint, angle, boundary_dist_1, boundary_dist_2,\
            stream_dist_1, stream_dist_2, interval_1, interval_2):
    """
    This function returns the ABC list of parallel flying path
    midpoint is the center point of stream polygon (after buffering)
    angle is the stream orientation
    boundary_dist_1 and boundary_dist_2 are the max positive and negative dists from boundary nodes
    to the center line of stream orientation
    stream_dist_1 and stream_dist_2 are the max positive and negative dists from stream nodes
    to the center line of stream orientation
    interval_1 and interval_2 are intervals in nearwater area and other area
    """
    A, B, C = getABCofAngle(angle, midpoint)
    positive_dist, negative_dist = 0, 0
    res_1, res_2 = [], []
    count = 0
    while positive_dist <= boundary_dist_1:
        interval = interval_1 if positive_dist < stream_dist_1 else interval_2
        if count == 0:
            cul_interval = copy.deepcopy(interval)
            count = count + 1
        else:
            cul_interval = cul_interval + interval
       # print([cul_interval, positive_dist, 1])
        new_A, new_B, new_C = moveCurrentLineByInterval(A, B, C, cul_interval)
        res_1.append([new_A, new_B, new_C])
        positive_dist += interval
    count = 0
    while negative_dist <= boundary_dist_2:
        interval = - interval_1 if negative_dist < stream_dist_2 else - interval_2
        if count == 0:
            cul_interval = copy.deepcopy(interval)
            count = count + 1
        else:
            cul_interval = cul_interval + interval
        
        new_A, new_B, new_C = moveCurrentLineByInterval(A, B, C, cul_interval)
        res_2.append([new_A, new_B, new_C])
        negative_dist -= interval
       # print([cul_interval, interval, negative_dist])
    return res_2[::-1] + [[A, B, C]] + res_1
    
def moveCurrentLineByInterval(A, B, C, delta):
    """
    This function returns the new A, B, C coefs if moving a line with A, B, C
    coefs by interval delta.

    """
    if A == 0 or B == 0:
        return A, B, C - delta
    else:
        k0 = - A / B
        b0 = - C / B
        db = delta * math.sqrt(1 + k0 ** 2) 
        b1 = b0 + db
        return k0, -1, b1
    
def getIntersectedSegmentsBetweenPolyandLine(boundary, A, B, C):
    """
    This function returns the intersection of polyline boundary to a line with A, B, C coefs
    """
    # Boundary should be 2-dimensional array, first column x, second column y, and looped. 
    res = []
    if boundary[0, 0] != boundary[-1, 0] or boundary[0, 1] != boundary[-1, 1]:
        new_boundary = np.vstack((boundary, np.array([boundary[0, 0],boundary[0, 1]])))
    else:
        new_boundary = copy.deepcopy(boundary)
    boundary_point_num = new_boundary.shape[0]
    if A == 0:
        x1, y1, x2, y2 = 0, - C/B, 1, - C/B
    elif B == 0:
        x1, y1, x2, y2 = - C/A, 0, - C/A, 0
    else:
        x1, y1, x2, y2 = 0, - C/B, - C/A, 0
    
    for i in range(boundary_point_num - 1):
        curr_x, curr_y = new_boundary[i, 0], new_boundary[i, 1]
        prev_x, prev_y = new_boundary[i + 1, 0], new_boundary[i + 1, 1]
        prev, curr = getSideofAPointOverALine(x1, y1, x2, y2, prev_x, prev_y),\
             getSideofAPointOverALine(x1, y1, x2, y2, curr_x, curr_y)
        if prev * curr < 0:
            res.append([prev_x, prev_y, curr_x, curr_y])
        elif curr == 0:
            res.append([curr_x, curr_y])
    return res

def getIntersectedSegmentsBetweenPolyandLines(boundary, ABC_list):
    """
    This function returns the intersection of polyline boundary to a sequence of parallel lines with A, B, C coefs

    """
    res_list = []
    for ABC in ABC_list:
        res_list.append(getIntersectedSegmentsBetweenPolyandLine(boundary, ABC[0], ABC[1], ABC[2]))
    return res_list

def refineResList(res_list, ABC_list):
    """
    This function refines intersection lists (if two endpoints, compute the intersected point)
    """
    point_list = []
    list_len = len(res_list)
    for i in range(list_len):
        if not res_list[i]:
            continue
        else:
            temp_res = res_list[i]
            temp_ABC = ABC_list[i]
            out = []
            for j in temp_res:
                if len(j) == 2:
                   out.append(j)
                else:
                   x, y = ComputeIntersection(j[0], j[1], j[2],\
                                              j[3], temp_ABC[0], temp_ABC[1], temp_ABC[2])
                   out.append([x, y])
            point_list.append(out)
    return point_list
    
def ComputeABCIntersection(A1, B1, C1, A2, B2, C2):
    """
    This function returns the coordinates of intersected points for two lines with a, b, c coefs
    """
    D = A1 * B2 - A2 * B1
    x = (B1 * C2 - B2 * C1) / D
    y = (- A1 * C2 + A2 * C1) / D
    return x, y

def ComputeIntersection(x1, y1, x2, y2, A, B, C):
    """
    This function returns the coordinates of intersected points for two lines with 
    x1,y1,x2,y2 coordinates and a, b, c coefs
    """
    A1, B1, C1 = computeABCofASegment(x1, y1, x2, y2)
    if A == 0:
        y = - C/B
        x = - (C1 + B1 * y) / A1
    elif B == 0:
        x = - C/A
        y = - (C1 + A1 * x) / B1
    else:
        x, y = ComputeABCIntersection(A1, B1, C1, A, B, C)
        return x, y    

def CorrectInitialDirectionOfParallelPoints(point_list):
    """
    This function changes the flight path from simply parallel to a zigzag pattern.

    """
    res = []
    ref = [point_list[0][1][0] - point_list[0][0][0], point_list[0][1][1] - point_list[0][0][1]]
    res.append(point_list[0])
    for i in range(1, len(point_list)):
        temp_direction = [point_list[i][1][0] - point_list[i][0][0], point_list[i][1][1] - point_list[i][0][1]]
        if ref[0] * temp_direction[0] + ref[1] * temp_direction[1] > 0:
            res.append(point_list[i])
        else:
            res.append(point_list[i][::-1])
    return res


def DetermineDirection(res_list, home_point):
    """
    This function returns the direction (from the first point of res_list[0] or the last one)
    """
    if len(res_list) // 2 == 0:
        # odd number, will no back
        option_1 = [res_list[0][0], res_list[-1][0]]
        option_2 = [res_list[0][-1], res_list[-1][-1]]
    else:
        option_1 = [res_list[0][0], res_list[-1][-1]]
        option_2 = [res_list[0][-1], res_list[-1][0]]
    dist_1 = get_distance_between_two_points(option_1[0][0], option_1[0][1], home_point[0], home_point[1]) + \
             get_distance_between_two_points(option_1[1][0], option_1[1][1], home_point[0], home_point[1])
    dist_2 = get_distance_between_two_points(option_2[0][0], option_2[0][1], home_point[0], home_point[1]) + \
             get_distance_between_two_points(option_2[1][0], option_2[1][1], home_point[0], home_point[1])
    if dist_1 > dist_2:
        return 1
    else:
        return -1

def CombineIntersectionPointList(res_list, start_direction):
    """
    This function stitches res_list to one list to a number
    """
    res = []
    curr_direction = copy.deepcopy(start_direction)
    point_pair_len = len(res_list)
    for i in range(point_pair_len):
        temp_point_pair = res_list[i]
        if not temp_point_pair:
            continue
        elif i == 0:
            if curr_direction == 1:
                res = res + temp_point_pair
            else:
                res = res + temp_point_pair[::-1]
        elif len(temp_point_pair) > 2:
            x_coords = [temp_point_pair[j][0] for j in range(len(temp_point_pair))];
            if len(x_coords) == len(list(set(x_coords))):
                min_ind = np.argmin(x_coords)
                max_ind = np.argmax(x_coords)
            else:
                y_coords = [temp_point_pair[j][1] for j in range(len(temp_point_pair))];
                min_ind = np.argmin(y_coords)
                max_ind = np.argmax(y_coords)
            prev_point = res[-1]
            
            curr_point_1 = temp_point_pair[min_ind]
            curr_point_2 = temp_point_pair[max_ind]
            dist_1 = get_distance_between_two_points(prev_point[0], prev_point[1], curr_point_1[0], curr_point_1[1])
            dist_2 = get_distance_between_two_points(prev_point[0], prev_point[1], curr_point_2[0], curr_point_2[1])
            if dist_1 <= dist_2:
                res = res + [curr_point_1, curr_point_2]
            else:
                res = res + [curr_point_2, curr_point_1]
        else:
            prev_point = res[-1]
            curr_point_1 = temp_point_pair[0]
            curr_point_2 = temp_point_pair[-1]
            dist_1 = get_distance_between_two_points(prev_point[0], prev_point[1], curr_point_1[0], curr_point_1[1])
            dist_2 = get_distance_between_two_points(prev_point[0], prev_point[1], curr_point_2[0], curr_point_2[1])
            if dist_1 <= dist_2:
                res = res + temp_point_pair
            else:
                res = res + temp_point_pair[::-1]
        #curr_direction = - curr_direction
    return res

def computeSegmentsFlyingTime(routes, speed_1, speed_2):
    dists = 0
    trans_dist = np.sqrt((routes[0][0][0] - routes[0][1][0]) ** 2 + \
                     (routes[0][0][1] - routes[0][1][1]) ** 2) + \
                     np.sqrt((routes[-1][0][0] - routes[-1][1][0]) ** 2 + \
                    (routes[-1][0][1] - routes[-1][1][1]) ** 2)
    for r in range(1, len(routes) - 1):
        x1, y1, x2, y2 = routes[r][0][0], routes[r][0][1], routes[r][1][0], routes[r][1][1]
        dists += np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    return trans_dist / speed_1, dists / speed_2

def computePointsFlyingTime(routes, speed_1, speed_2):
    dists = 0
    trans_dist = np.sqrt((routes[0][0] - routes[1][0]) ** 2 + \
                     (routes[0][1] - routes[1][1]) ** 2) + \
                     np.sqrt((routes[-1][0] - routes[-2][0]) ** 2 + \
                    (routes[-1][1] - routes[-2][1]) ** 2)
    for r in range(1, len(routes) - 1):
        x1, y1, x2, y2 = routes[r][0], routes[r][1], routes[r + 1][0], routes[r + 1][1]
        dists += np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    return trans_dist / speed_1, dists / speed_2
    
def SamplingPoint1(routes, interval_1, interval_2):
    # Compute the total length of the route by connecting all the segments
    total_length = 0
    camera_points, dxs, dys = [routes[0][0]], \
                    [routes[0][1][0] - routes[0][0][0]], [routes[0][1][1] - routes[0][0][1]]
    curr_segment = routes[0]
    start_point, end_point = curr_segment[0], curr_segment[1]
    seg_dist = ((end_point[0]-start_point[0])**2 + (end_point[1]-start_point[1])**2)**0.5
    remain_dist = seg_dist
    while remain_dist >= interval_1:
        remain_dist = remain_dist - interval_1
        ratio = remain_dist / seg_dist
        dx, dy = (end_point[0]-start_point[0]) * ratio, (end_point[1]-start_point[1]) * ratio
        camera_points.append([end_point[0] - dx, end_point[1] - dy])
        dxs.append(end_point[0]-start_point[0])
        dys.append(end_point[1]-start_point[1])
        
    if remain_dist != 0:
        camera_points.append(routes[0][1])
        dxs.append(routes[0][1][0] - routes[0][0][0])
        dys.append(routes[0][1][1] - routes[0][0][1])
        remain_dist = 0
        
    for i in range(1, len(routes) - 1):
        curr_segment = routes[i]
        start_point, end_point = curr_segment[0], curr_segment[1]
        seg_dist = ((end_point[0]-start_point[0])**2 + (end_point[1]-start_point[1])**2)**0.5
        remain_dist = remain_dist + seg_dist
        while remain_dist >= interval_2:
            remain_dist = remain_dist - interval_2
            ratio = remain_dist / seg_dist
            dx, dy = (end_point[0]-start_point[0]) * ratio, (end_point[1]-start_point[1]) * ratio
            camera_points.append([end_point[0] - dx, end_point[1] - dy])
            dxs.append(end_point[0]-start_point[0])
            dys.append(end_point[1]-start_point[1])

    camera_points.append(routes[len(routes) - 2][1])
    dxs.append(routes[len(routes) - 2][1][0] - routes[len(routes) - 2][0][0])
    dys.append(routes[len(routes) - 2][1][1] - routes[len(routes) - 2][0][1])
        
    curr_segment = routes[len(routes) - 1]
    start_point, end_point = curr_segment[0], curr_segment[1]
    seg_dist = ((end_point[0]-start_point[0])**2 + (end_point[1]-start_point[1])**2)**0.5
    remain_dist = seg_dist
    while remain_dist >= interval_1:
        remain_dist = remain_dist - interval_1
        ratio = remain_dist / seg_dist
        dx, dy = (end_point[0]-start_point[0]) * ratio, (end_point[1]-start_point[1]) * ratio
        camera_points.append([end_point[0] - dx, end_point[1] - dy])
        dxs.append(end_point[0]-start_point[0])
        dys.append(end_point[1]-start_point[1])
    if remain_dist != 0:
        camera_points.append(routes[len(routes) - 1][1])
        dxs.append(routes[len(routes) - 1][1][0] - routes[len(routes) - 1][0][0])
        dys.append(routes[len(routes) - 1][1][1] - routes[len(routes) - 1][0][1])
        remain_dist = 0
    return camera_points, dxs, dys

def is_point_in_polygon(point, polygon):
    num_vertices = len(polygon)
    if num_vertices < 3:
        return False

    # Create a line segment from the point to a point outside the polygon.
    outside_point = [max([v[0] for v in polygon]) + 1, point[1]]
    segment = [point, outside_point]

    # Count the number of times the line segment intersects the edges of the polygon.
    num_intersections = 0
    for i in range(num_vertices):
        edge_start = polygon[i]
        edge_end = polygon[(i + 1) % num_vertices]

        if segment_intersects_edge(segment, edge_start, edge_end):
            num_intersections += 1

    # If the number of intersections is odd, the point is inside the polygon.
    return num_intersections % 2 == 1


def segment_intersects_edge(segment, edge_start, edge_end):
    # Get the x and y coordinates of the segment endpoints and the edge endpoints.
    segment_x1, segment_y1 = segment[0]
    segment_x2, segment_y2 = segment[1]
    edge_x1, edge_y1 = edge_start
    edge_x2, edge_y2 = edge_end

    # Calculate the direction of the line segment and the edge.
    segment_dx = segment_x2 - segment_x1
    segment_dy = segment_y2 - segment_y1
    edge_dx = edge_x2 - edge_x1
    edge_dy = edge_y2 - edge_y1

    # Calculate the determinant of the 2x2 matrix formed by the direction vectors.
    determinant = segment_dx * edge_dy - segment_dy * edge_dx

    # If the determinant is 0, the lines are parallel and do not intersect.
    if determinant == 0:
        return False

    # Calculate the intersection point of the lines.
    t = ((edge_x1 - segment_x1) * edge_dy - (edge_y1 - segment_y1) * edge_dx) / determinant
    u = ((edge_x1 - segment_x1) * segment_dy - (edge_y1 - segment_y1) * segment_dx) / determinant

    # If the intersection point is within the bounds of the segment and the edge, they intersect.
    if 0 <= t <= 1 and 0 <= u <= 1:
        return True

    return False

def find_four_square_coords(curr_x, curr_y, real_dx, real_dy):
    '''
    curr_x and curr_y are the x and y coordinates of square center
    real_dx and real_dy the coordinate difference between square center and the midpoint of an edge
    '''
    point_1, point_2, point_3, point_4 = [curr_x + real_dx, curr_y + real_dy], \
             [curr_x - real_dy, curr_y + real_dx], [curr_x - real_dx, curr_y - real_dy], \
             [curr_x + real_dy, curr_y - real_dx]
    #print([point_1])
    #print([point_2])
    #print([point_3])
    #print([point_4])
    new_point_1, new_point_2, new_point_3, new_point_4 = \
             [point_1[0] - real_dy, point_1[1] + real_dx], [point_3[0] - real_dy, point_3[1] + real_dx],\
             [point_3[0] + real_dy, point_3[1] - real_dx], [point_1[0] + real_dy, point_1[1] - real_dx]
    points = np.array([new_point_1, new_point_2, new_point_3, new_point_4])
    return points

def computeCameraCoverageMatrix(camera_points, dxs, dys,  hori_dist_test, row_num_a1, col_num_a1):
    empty_img_a1 = np.zeros((row_num_a1, col_num_a1, 3))
    count_matrix = np.zeros((row_num_a1, col_num_a1))
    for i in range(len(camera_points)):
        curr_x, curr_y = camera_points[i][0], camera_points[i][1]
        temp_dx, temp_dy = dxs[i], dys[i]
        temp_dist = np.sqrt(temp_dx ** 2 + temp_dy ** 2)
        scale = 0.5 * hori_dist_test / temp_dist
        real_dx, real_dy = temp_dx * scale, temp_dy * scale
        points = find_four_square_coords(\
                                    curr_x, curr_y, real_dx, real_dy)
        temp_img = cv2.fillPoly(copy.deepcopy(empty_img_a1), \
                            pts = np.int32([points]), color =(255,255,255))
        count_matrix = count_matrix + np.where(temp_img[:, :, 0] > 0, 1, 0)
    return count_matrix
    

def IfStraightLineCoveredBlockedArea(blocked_map, x1, y1, x2, y2):
    covered_pixels = FloydSmoothing.ExtractCoveredPixels(x1, y1, x2, y2)
    new_covered_pixels = np.array([covered_pixels[1, :], covered_pixels[0, :]])
    #print(covered_pixels)
    return FloydSmoothing.CheckStraightLineWorkable(new_covered_pixels, blocked_map) 

def SeperateStraightAndDetourRoute(route, blocked_map):
    is_detour_list = []
    for i in range(len(route) - 1):
        # print(i)
        x1, y1, x2, y2 = route[i][0], route[i][1], route[i + 1][0], route[i + 1][1]
        if IfStraightLineCoveredBlockedArea(blocked_map, x1, y1, x2, y2):
            is_detour_list.append(True)
        else:
            is_detour_list.append(False)
    return is_detour_list

def UpdateStartAndEndPoint(s_x, s_y, g_x, g_y, blocked_map, min_interval, n = 40):
    dist = np.sqrt((s_x - g_x) ** 2 + (s_y - g_y) ** 2)
    number = n if dist / min_interval > n else int(dist / min_interval)
    delta_x = (g_x - s_x) / number
    delta_y = (g_y - s_y) / number
    curr_s_x, curr_s_y, curr_g_x, curr_g_y = copy.deepcopy(s_x), copy.deepcopy(s_y),\
        copy.deepcopy(g_x), copy.deepcopy(g_y)
    while blocked_map[int(curr_s_x), int(curr_s_y)] != 0:
        curr_s_x, curr_s_y = curr_s_x + delta_x, curr_s_y + delta_y
    while blocked_map[int(curr_g_x), int(curr_g_y)] != 0 or abs(curr_s_x - curr_g_x) + abs(curr_s_x - curr_g_x) <= 0.1:
        curr_g_x, curr_g_y = curr_g_x - delta_x, curr_g_y - delta_y
    return [[curr_s_x, curr_s_y], [curr_g_x, curr_g_y]]

def ComputeSegmentLens(routes):
    lens = [np.sqrt((routes[r][0] - routes[r + 1][1]) ** 2 + (routes[r][1] - routes[r + 1][1]) ** 2) for r in range(len(routes) - 1)]
    return lens
        
def getSegmentLenThresh(lens):
    return max(np.median(lens), min(min(lens) * 1.5, max(lens) * 0.9))

def updateRouteSegments(blocked_map, segments, interval_num):
    new_segments = []
    last_end_node = None

    for i, (start_node, end_node) in enumerate(segments):
        if last_end_node is not None and last_end_node != start_node:
            start_node = last_end_node

        if blocked_map[int(start_node[1])][int(start_node[0])] == 1:
            dx = (end_node[0] - start_node[0]) / interval_num
            dy = (end_node[1] - start_node[1]) / interval_num

            for j in range(1, interval_num + 1):
                new_x = start_node[0] + j * dx
                new_y = start_node[1] + j * dy

                if blocked_map[int(new_y)][int(new_x)] == 0:
                    start_node = [new_x, new_y]
                    break

        if blocked_map[int(end_node[1])][int(end_node[0])] == 1:
            dx = (start_node[0] - end_node[0]) / interval_num
            dy = (start_node[1] - end_node[1]) / interval_num

            for j in range(1, interval_num + 1):
                new_x = end_node[0] + j * dx
                new_y = end_node[1] + j * dy

                if blocked_map[int(new_y)][int(new_x)] == 0:
                    end_node = [new_x, new_y]
                    break

        new_segments.append([start_node, end_node])
        last_end_node = end_node

    return new_segments

def bufferAreaBoundary(boundary, rows, cols, buffer_dist):
    boundary = np.array(boundary)
    centroid = np.mean(boundary, axis=0)
    vectors = boundary - centroid
    magnitudes = np.sqrt(np.sum(vectors ** 2, axis=1))
    vectors /= magnitudes[:, np.newaxis]
    new_boundary = boundary + buffer_dist * vectors
    new_boundary[:, 0] = np.clip(new_boundary[:, 0], 0, cols - 1)
    new_boundary[:, 1] = np.clip(new_boundary[:, 1], 0, rows - 1)
    new_boundary = new_boundary.tolist()
    return new_boundary

def intersects_blocked_positions(blocked_map, x1, y1, x2, y2):
    delta_x = x2 - x1
    delta_y = y2 - y1
    steps = max(abs(delta_x), abs(delta_y))

    for step in range(int(steps)):
        t = step / steps
        x = x1 + t * delta_x
        y = y1 + t * delta_y
        if blocked_map[int(y), int(x)] == float('inf'):
            return True
    return False

def splitDetouredRoute(blocked_map, start_x, start_y, end_x, end_y, n):

    x_step = (end_x - start_x) / n
    y_step = (end_y - start_y) / n
    #print([x_step, y_step])
    detour_start_point = [start_x, start_y]
    for i in range(1, n):
        
        current_x = start_x + i * x_step
        current_y = start_y + i * y_step
        prev_x = start_x + (i - 1) * x_step
        prev_y = start_y + (i - 1) * y_step
        #print([i, current_x, current_y, int(current_y), int(current_x), prev_x, prev_y])
        if current_x == prev_x and current_y == prev_y:
            continue
        elif blocked_map[int(current_y), int(current_x)] == float('inf'):
            #print(blocked_map[int(current_y), int(current_x)])
            break
        elif intersects_blocked_positions(blocked_map, start_x, start_y, current_x, current_y):
            break
        detour_start_point = [int(current_x - x_step), int(current_y - y_step)]

    detour_end_point = [end_x, end_y]
    
    for j in range(n - 1, i, -1):
        current_x = start_x + j * x_step
        current_y = start_y + j * y_step
        prev_x = start_x + (j + 1) * x_step
        prev_y = start_y + (j + 1) * y_step
        #print([j, current_x, current_y, int(current_y), int(current_x), prev_x, prev_y])
        if current_x == prev_x and current_y == prev_y:
            continue
        elif blocked_map[int(current_y), int(current_x)] == float('inf'):
            #print(blocked_map[int(current_y), int(current_x)])
            break
        elif intersects_blocked_positions(blocked_map, end_x, end_y, current_x, current_y):  
            break
        detour_end_point = [int(current_x + x_step), int(current_y + y_step)]
    return [[start_x, start_y], detour_start_point, detour_end_point, [end_x, end_y]]

def SeperateStraightAndDetourSegment(segments, blocked_map):
    def intersects_blocked(pos):
        x, y = pos
        if x < 0 or y < 0 or x >= len(blocked_map[0]) or y >= len(blocked_map):
            return True  # position is out of bounds
        return blocked_map[y][x] == float('inf')  # position is blocked

    def intersects_segment(segment):
        start, end = segment
        x0, y0 = start
        x1, y1 = end
        dx = x1 - x0
        dy = y1 - y0
        steps = max(abs(dx), abs(dy))
        x_step = dx / steps if steps > 0 else 0
        y_step = dy / steps if steps > 0 else 0
        x = x0
        y = y0
        for i in range(int(steps) + 1):
            if intersects_blocked((int(x), int(y))):
                return True
            x += x_step
            y += y_step
        return False

    SeperateRes = [intersects_segment(segment) for segment in segments]
    return SeperateRes    

def DetourAndSmoothForSegments(subtracted_segments, is_detour_list, blocked_matrix):
    res_1, res_2 = [], []
    for i in range(len(subtracted_segments)):
        if is_detour_list[i]:
            s_x, s_y, g_x, g_y = int(np.floor(subtracted_segments[i][0][0])), int(np.floor(subtracted_segments[i][0][1])), \
                                     int(np.floor(subtracted_segments[i][1][0])), int(np.floor(subtracted_segments[i][1][1]))
            if not intersects_blocked_positions(blocked_matrix, s_x, s_y, g_x, g_y):
                res_1.append(subtracted_segments[i])
                res_2.append(subtracted_segments[i])
                continue
            new_nodes = splitDetouredRoute(blocked_matrix, s_x, s_y, \
                                                                        g_x, g_y, 40)
            new_s_x, new_s_y, new_g_x, new_g_y = new_nodes[1][0], new_nodes[1][1], new_nodes[2][0], new_nodes[2][1]
            # print(i)
            # print([new_s_x, new_s_y, new_g_x, new_g_y])
            nodes_x, nodes_y = d_star_lite_optimized.DLiteMain(blocked_matrix, int(new_g_y), int(new_g_x), int(new_s_y), int(new_s_x))
            
            if nodes_x[0] == new_nodes[0][1] and nodes_y[0] == new_nodes[0][0]:
                nodes_x.pop(0)
                nodes_y.pop(0)
            if nodes_x[-1] == new_nodes[-1][1] and nodes_y[-1] == new_nodes[-1][0]:
                nodes_x.pop(-1)
                nodes_y.pop(-1)
            if nodes_x:
                if nodes_x[0] != new_s_y or nodes_y[0] != new_s_x:
                    nodes_x = [new_s_y] + nodes_x
                    nodes_y = [new_s_x] + nodes_y
                if nodes_x[-1] != new_g_y or nodes_y[-1] != new_g_x:
                    nodes_x = nodes_x + [new_g_y]
                    nodes_y = nodes_y + [new_g_x]
            
            if not nodes_x:
                nodes_x = [new_nodes[0][1]] + nodes_x + [new_nodes[-1][1]]
                nodes_y = [new_nodes[0][0]] + nodes_y + [new_nodes[-1][0]]
            
            
            new_path = FloydSmoothing.FloydSmoothing(np.transpose(np.array([nodes_x, nodes_y])), blocked_matrix)
            new_path = FloydSmoothing.FloydSmoothing(new_path, blocked_matrix)
            new_path = new_path.tolist()
            
            if i != 0:
                temp_dist = abs((np.sqrt(nodes_x[0]) - subtracted_segments[i - 1][1][1]) ** 2 + \
                               (nodes_y[0] - subtracted_segments[i - 1][1][0])** 2)
                if temp_dist <= 1.5:
                    nodes_x[0] = subtracted_segments[i - 1][1][1]
                    nodes_y[0] = subtracted_segments[i - 1][1][0]
                    new_path[0][0] = subtracted_segments[i - 1][1][1]
                    new_path[0][1] = subtracted_segments[i - 1][1][0]
                else:
                    nodes_x = [subtracted_segments[i - 1][1][1]] + nodes_x
                    nodes_y = [subtracted_segments[i - 1][1][0]] + nodes_y
                    new_path = [[subtracted_segments[i - 1][1][1], subtracted_segments[i - 1][1][0]]] + new_path
            
            if i != len(subtracted_segments) - 1:
                temp_dist = abs((np.sqrt(nodes_x[-1]) - subtracted_segments[i + 1][0][1]) ** 2 + \
                               (nodes_y[-1] - subtracted_segments[i + 1][0][0])** 2)
                if temp_dist <= 1.5:
                    nodes_x[-1] = subtracted_segments[i + 1][0][1]
                    nodes_y[-1] = subtracted_segments[i + 1][0][0]
                    new_path[-1][0] = subtracted_segments[i + 1][0][1]
                    new_path[-1][1] = subtracted_segments[i + 1][0][0]
                else:
                    nodes_x = nodes_x + [subtracted_segments[i + 1][0][1]]
                    nodes_y = nodes_y + [subtracted_segments[i + 1][0][0]]
                    new_path = new_path + [[subtracted_segments[i + 1][0][1], subtracted_segments[i + 1][0][0]]]
            
            for j in range(len(nodes_x) - 1):
                res_1.append([[nodes_y[j], nodes_x[j]], [nodes_y[j + 1], nodes_x[j + 1]]])
            
            for j in range(len(new_path) - 1):
                res_2.append([[new_path[j][1], new_path[j][0]], [new_path[j + 1][1], new_path[j + 1][0]]])
        else:
            res_1.append(subtracted_segments[i])
            res_2.append(subtracted_segments[i])
    return res_1, res_2

def RemoveRoundingSegments(res_1):
    for r1 in range(len(res_1) - 1, -1, -1):
        temp_1 = res_1[r1]
        if np.sqrt((temp_1[0][0] - temp_1[1][0]) ** 2 + (temp_1[0][1] - temp_1[1][1]) ** 2) < 0.99:
            if r1 == len(res_1) - 1 or r1 == 0:
                res_1.pop(r1)
            else:
                res_1[r1 - 1][1] = res_1[r1][1]
                res_1.pop(r1)
    return res_1

def ExtractMinMaxVals(boundary, buffered_boundary, segments):
    boundary = np.array(boundary)
    segments = np.array(segments)
    buffered_boundary = np.array(buffered_boundary)
    min_x = min(boundary[:, 0].min(), int(buffered_boundary[:, 0].min()), segments[:, :, 0].min())
    min_y = min(boundary[:, 1].min(), int(buffered_boundary[:, 1].min()), segments[:, :, 1].min())
    max_x = max(boundary[:, 0].max(), int(buffered_boundary[:, 0].max()), segments[:, :, 0].max())
    max_y = max(boundary[:, 1].max(), int(buffered_boundary[:, 1].max()), segments[:, :, 1].max())
    return min_x, min_y, max_x, max_y

def UpdateCoordsBySubtractingMins(coords, min_x, min_y):
    coords = np.array(coords) - np.array([min_x, min_y])
    return coords.tolist()

def UpdateCoordsByAddingMins(coords, min_x, min_y):
    coords = np.array(coords) + np.array([min_x, min_y])
    return coords.tolist()

def PlotFlyingPathsBeforeDetour(blocked_img, segments, color, thickness):
    for r in range(len(segments)):
        x1, y1, x2, y2 = segments[r][0][0], segments[r][0][1], segments[r][1][0], segments[r][1][1]
        if r == 0:
            test_img = cv2.line(blocked_img, (int(x1), int(y1)), (int(x2), int(y2)), \
                   color = color, thickness = thickness)
        else:
            test_img = cv2.line(test_img, (int(x1), int(y1)), (int(x2), int(y2)), \
                   color = color, thickness = thickness)
    return test_img
        
def PlotFlyingPathsAfterDetour(blocked_img, segments, color, thickness):
    for r in range(len(segments)):
        curr_len = len(segments[r])
        for c in range(curr_len - 1):
            x1, y1, x2, y2 = segments[r][c][0], segments[r][c][1], segments[r][c + 1][0], segments[r][c + 1][1]
        
            if r == 0 and c == 0:
                test_img = cv2.line(blocked_img, (int(x1), int(y1)), (int(x2), int(y2)), \
                       color = color, thickness = thickness)
            else:
                test_img = cv2.line(test_img, (int(x1), int(y1)), (int(x2), int(y2)), \
                       color = color, thickness = thickness)
    return test_img

def PlotFlyingPathsUsingPlt(blocked_img, boundary, buffered_boundary, segments, color1, color2, thickness1, thickness2):
    boundary = np.array(boundary)
    segments = np.array(segments)
    buffered_boundary = np.array(buffered_boundary)

    min_x = min(boundary[:, 0].min(), int(buffered_boundary[:, 0].min()), segments[:, :, 0].min())
    min_y = min(boundary[:, 1].min(), int(buffered_boundary[:, 1].min()), segments[:, :, 1].min())
    max_x = max(boundary[:, 0].max(), int(buffered_boundary[:, 0].max()), segments[:, :, 0].max())
    max_y = max(boundary[:, 1].max(), int(buffered_boundary[:, 1].max()), segments[:, :, 1].max())
    print([min_x, min_y, max_x, max_y])

    new_img = blocked_img[min_y:max_y+1, min_x:max_x+1]

    new_boundary = boundary - np.array([min_x, min_y])
    new_segments = segments - np.array([min_x, min_y])
    
    ax = plt.gca()
    plt.imshow(new_img, cmap='gray', origin='lower', extent=(0, max_x-min_x, 0, max_y-min_y))

    plt.plot(np.append(new_boundary[:, 0], new_boundary[0, 0]), np.append(new_boundary[:, 1], new_boundary[0, 1]), color=color1, linewidth=thickness1)

    for segment in new_segments:
        plt.plot(segment[:, 0], segment[:, 1], color=color2, linewidth=thickness2)
    
    plt.axis('equal')
    ax.invert_yaxis()
    plt.show()
    
def PlotFlyingPathsUsingPltWithPhaseMarking(new_img, boundary, segments, color1, color2, color3, color4, thickness1, thickness2, \
                                            home_marker_style, home_marker_color, home_marker_size):
    boundary = np.array(boundary)
    segments = np.array(segments)

    ax = plt.gca()
    plt.imshow(new_img, cmap='gray', origin='lower')

    plt.plot(np.append(boundary[:, 0], boundary[0, 0]), np.append(boundary[:, 1], boundary[0, 1]), color = color1, linewidth = thickness1)

    for s in range(len(segments)):
        segment = segments[s]
        if s == 0:
            plt.plot(segment[:, 0], segment[:, 1], color = color2, linewidth = thickness2)
        elif s == len(segments) - 1:
            plt.plot(segment[:, 0], segment[:, 1], color = color4, linewidth = thickness2)
        else:
            plt.plot(segment[:, 0], segment[:, 1], color = color3, linewidth = thickness2)
    plt.scatter(segments[0, 0, 0], segments[0, 0, 1], c = home_marker_color, \
                s = home_marker_size, marker = home_marker_style)
    plt.axis('off')
    ax.invert_yaxis()
    plt.show()

def ConvertImgColor(input_img, color_1, color_2, color_3=(255, 255, 255)):
    # Convert the image to grayscale
    gray = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    
    # Threshold the grayscale image to create a binary mask
    ret, mask = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    
    # Create two new images, one for color_1 and one for color_2
    color_1_img = np.zeros_like(input_img)
    color_2_img = np.zeros_like(input_img)
    
    # Set the pixel values for color_1 and color_2
    color_1_img[:] = color_1
    color_2_img[:] = color_2
    
    # Convert the colors to the same data type as the input image
    color_1_img = color_1_img.astype(input_img.dtype)
    color_2_img = color_2_img.astype(input_img.dtype)
    
    # Apply the colors to the black-and-white image
    color_1_pixels = cv2.bitwise_and(color_1_img, color_1_img, mask=mask)
    color_2_pixels = cv2.bitwise_and(color_2_img, color_2_img, mask=cv2.bitwise_not(mask))
    
    # Create a mask for the border pixels
    border_mask = np.zeros((input_img.shape[0], input_img.shape[1]), dtype=np.uint8)
    border_mask[0,:] = 255
    border_mask[-1,:] = 255
    border_mask[:,0] = 255
    border_mask[:,-1] = 255
    
    # Create a new image for color_3 and set the pixel values
    color_3_img = np.zeros_like(input_img)
    color_3_img[:] = color_3
    
    # Apply the color to the border pixels
    color_3_pixels = cv2.bitwise_and(color_3_img, color_3_img, mask=border_mask)
    
    # Add the three color images together to get the final result
    result = cv2.add(color_1_pixels, color_2_pixels)
    result = cv2.add(result, color_3_pixels)
    
    return result

def ConvertImgColor(input_img, color_1, color_2, color_3=(255, 255, 255)):
    # Convert the image to grayscale
    gray = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    
    # Threshold the grayscale image to create a binary mask
    ret, mask = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    
    # Create two new images, one for color_1 and one for color_2
    color_1_img = np.zeros_like(input_img)
    color_2_img = np.zeros_like(input_img)
    
    # Set the pixel values for color_1 and color_2
    color_1_img[:] = color_1
    color_2_img[:] = color_2
    
    # Convert the colors to the same data type as the input image
    color_1_img = color_1_img.astype(input_img.dtype)
    color_2_img = color_2_img.astype(input_img.dtype)
    
    # Apply the colors to the black-and-white image
    color_1_pixels = cv2.bitwise_and(color_1_img, color_1_img, mask=mask)
    color_2_pixels = cv2.bitwise_and(color_2_img, color_2_img, mask=cv2.bitwise_not(mask))
    
    # Create a mask for the border pixels
    border_mask = np.zeros((input_img.shape[0], input_img.shape[1]), dtype=np.uint8)
    border_mask[0,:] = 255
    border_mask[-1,:] = 255
    border_mask[:,0] = 255
    border_mask[:,-1] = 255
    
    # Create a new image for color_3 and set the pixel values
    color_3_img = np.zeros_like(input_img)
    color_3_img[:] = color_3
    
    # Apply the color to the border pixels
    color_3_pixels = cv2.bitwise_and(color_3_img, color_3_img, mask=border_mask)
    
    # Add the three color images together to get the final result
    result = cv2.add(color_1_pixels, color_2_pixels)
    result = cv2.add(result, color_3_pixels)
    
    return result