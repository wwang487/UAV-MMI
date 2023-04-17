# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 02:05:39 2022

@author: wei wang
"""
import cv2
import os
import numpy as np
import copy
from osgeo import gdal, osr

def CheckIfFolderExist(filepath):
    if os.path.exists(f'{filepath}'):
        pass
    else:
        os.mkdir(f'{filepath}') 
        
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

def get_tif_info(tif_path):
    if tif_path.endswith('.tif') or tif_path.endswith('.TIF'):
        dataset = gdal.Open(tif_path)
        pcs = osr.SpatialReference()
        pcs.ImportFromWkt(dataset.GetProjection())
        gcs = pcs.CloneGeogCS()
        extend = dataset.GetGeoTransform()
        # im_width = dataset.RasterXSize #栅格矩阵的列数
        # im_height = dataset.RasterYSize #栅格矩阵的行数
        shape = (dataset.RasterYSize, dataset.RasterXSize)
    else:
        raise "Unsupported file format"

    img = dataset.GetRasterBand(1).ReadAsArray()  # (height, width)
    # img(ndarray), gdal数据集、地理空间坐标系、投影坐标系、栅格影像大小
    return img, dataset, gcs, pcs, extend, shape

def longlat_to_xy(gcs, pcs, lon, lat):
    ct = osr.CoordinateTransformation(gcs, pcs)
    coordinates = ct.TransformPoint(lon, lat)
    return coordinates[0], coordinates[1], coordinates[2]


def xy_to_lonlat(gcs, pcs, x, y):
    ct = osr.CoordinateTransformation(gcs, pcs)
    lon, lat, _ = ct.TransformPoint(x, y)
    return lon, lat


def xy_to_rowcol(extend, x, y):
    a = np.array([[extend[1], extend[2]], [extend[4], extend[5]]])
    b = np.array([x - extend[0], y - extend[3]])

    row_col = np.linalg.solve(a, b)
    row = int(np.floor(row_col[1]))
    col = int(np.floor(row_col[0]))

    return row, col


def rowcol_to_xy(extend, row, col):
    x = extend[0] + col * extend[1] + row * extend[2]
    y = extend[3] + col * extend[4] + row * extend[5]
    return x, y

def get_value_by_coordinates(img, dataset, gcs, pcs, extend, shape, coordinates, coordinate_type='rowcol'):

    if coordinate_type == 'rowcol':
        value = img[coordinates[0], coordinates[1]]
    elif coordinate_type == 'lonlat':
        x, y, _ = longlat_to_xy(gcs, pcs, coordinates[0], coordinates[1])
        row, col = xy_to_rowcol(extend, x, y)
        value = img[row, col]
    elif coordinate_type == 'xy':
        row, col = xy_to_rowcol(extend, coordinates[0], coordinates[1])
        value = img[row, col]
    else:
        raise 'coordinated_type error'
    return value

def adjustPointCoordsWithTopLeftPoint(point_x, point_y, tl_x, tl_y):
    return point_x - tl_x, point_y - tl_y

def adjustPointListCoordsWithTopLeftPoint(point_x_list, point_y_list, tl_x, tl_y):
    res_x, res_y = [], []
    for i in range(len(point_x_list)):
        temp_x, temp_y = adjustPointCoordsWithTopLeftPoint(point_x_list[i], \
                                                point_y_list[i], tl_x, tl_y)
        res_x.append(temp_x)
        res_y.append(temp_y)
    return res_x, res_y

def ReadXYCoordsFromTXTFile(file_folder, file_name):
    file_path = file_folder + file_name
    X, Y = [], []
    with open(file_path, "r") as f:
        lines = f.readlines()
    f.close()
    line_num = len(lines)
    for i in range(line_num):
        if i == 0:
            continue
        else:
            line_split = lines[i].split(',')
            
            X.append(float(line_split[-2]))
            Y.append(float(line_split[-1]))
    return X, Y

def DetermineTheTOPLEFTCoordsOfATif(tif_path):
    dataset = gdal.Open(tif_path)
    getSRSPair(dataset)
    x0, y0 = imagexy2geo(dataset, 0, 0)
    shape = (dataset.RasterYSize, dataset.RasterXSize)
    x1, y1 = imagexy2geo(dataset, shape[0] - 1, shape[1] - 1)
    return x0, y0, x1, y1, shape[0], shape[1]

def GetImgVersionOfATif(tif_path):
    dataset = gdal.Open(tif_path)
    return dataset.GetRasterBand(1).ReadAsArray()

def ChangeImgToSimpleBWImg(img):
    row_num, col_num = img.shape[:2]
    res_img = np.zeros((row_num, col_num))
    if len(img.shape) == 2:
        res_img = copy.deepcopy(img)
    else:
        res_img = img[:,:,1] + img[:,:,2], img[:,:,3]
    return np.uint8(np.where(res_img > 0, 255, 0))

def ConvertXYCoordsToImgCoords(X, Y, x0, y0, x1, y1, x_num, y_num):
    delta_x = (x1 - x0) / x_num
    delta_y = (y1 - y0) / y_num
    X_coords, Y_coords = [], []
    for i in range(len(X)):
        X_coords.append((X[i] - x0) / delta_x)
        Y_coords.append((Y[i] - y0) / delta_y)
    return X_coords, Y_coords    

def CombineBlockedImg(init_img, boundary_X_coords, boundary_Y_coords, row_num, col_num):
    white_img = np.uint8(np.ones((row_num, col_num, 3))* 255)
    empty_img = np.uint8(np.zeros((row_num, col_num, 3)))
    boundary = np.int32(np.transpose(np.array([boundary_X_coords, boundary_Y_coords])))
    boundary_img = cv2.fillPoly(copy.deepcopy(empty_img), pts = [boundary], color=(255, 255, 255))

    boundary_inverse_img = white_img - boundary_img
    final_blocked_img = init_img + boundary_inverse_img
    return final_blocked_img
    
if __name__ == "__main__":
    tif_path = './Tif_Input/blocked1.tif'
    txt_file_folder = './Shp_Input/'
    area_1_name = 'Area1_Point.txt'
    
