import math
import os
import sys
import pdb
import cv2
import numpy as np

src_dir = os.path.dirname(os.path.realpath(__file__))
while not src_dir.endswith("sfa"):
    src_dir = os.path.dirname(src_dir)
if src_dir not in sys.path:
    sys.path.append(src_dir)

import config.kitti_config as cnf

# 通过点云数据pointcloud包括(x,y,z，反射强度)来计算鸟瞰图BEV
# 返回的图像由每个(x, y)点对应的高度，反射强度，密度三个图构成最终伪RGB图片
def makeBEVMap(PointCloud_, boundary):
    Height = cnf.BEV_HEIGHT + 1
    Width = cnf.BEV_WIDTH + 1

    # Discretize Feature Map 离散化特征图，将x,y原始雷达数据除以边界值再乘以BEV的尺寸以获得BEV尺度下的X，Y坐标
    PointCloud = np.copy(PointCloud_)
    PointCloud[:, 0] = np.int_(np.floor(PointCloud[:, 0] / cnf.DISCRETIZATION)) # 对x离散化
    PointCloud[:, 1] = np.int_(np.floor(PointCloud[:, 1] / cnf.DISCRETIZATION) + Width / 2)
    # sort-3times 在x, y值相等的情况下，将Z值从高到低返回索引
    sorted_indices = np.lexsort((-PointCloud[:, 2], PointCloud[:, 1], PointCloud[:, 0]))
    PointCloud = PointCloud[sorted_indices]

    # 去除点云中x,y维度下重复的数据，返回清除并排序好的只包含（x,y）点云数据，在unique中的数据的原始下标，和重复次数。
    _, unique_indices, unique_counts = np.unique(PointCloud[:, 0:2], axis=0, return_index=True, return_counts=True)
    # 根据上面计算好的下标再次整理数据，只保留点云中非重复的数据
    PointCloud_top = PointCloud[unique_indices]
    # Height Map, Intensity Map & Density Map 高度图，反射强度图，点云密度图
    heightMap = np.zeros((Height, Width)) # (608 608)
    intensityMap = np.zeros((Height, Width))
    densityMap = np.zeros((Height, Width))

    # some important problem is image coordinate is (y,x), not (x,y)
    # print(np.int_(PointCloud_top[:, 0]))
    max_height = float(np.abs(boundary['maxZ'] - boundary['minZ']))
    normalizedCounts = np.minimum(1.0, np.log(unique_counts + 1) / np.log(64))  # 把统计的次数归一化
    
    heightMap[np.int_(PointCloud_top[:, 0]), np.int_(PointCloud_top[:, 1])] = PointCloud_top[:, 2] / max_height # 每个x,y坐标点对应一个z高度，并且将其归一化生成高度图
    intensityMap[np.int_(PointCloud_top[:, 0]), np.int_(PointCloud_top[:, 1])] = PointCloud_top[:, 3] # 生成反射强度图
    densityMap[np.int_(PointCloud_top[:, 0]), np.int_(PointCloud_top[:, 1])] = normalizedCounts  # 生成点云密度图

    # RGB三个通道分别代表了高度通道，反射强度通道，点密度通道
    RGB_Map = np.zeros((3, Height - 1, Width - 1))
    RGB_Map[2, :, :] = densityMap[:cnf.BEV_HEIGHT, :cnf.BEV_WIDTH]  # r_map
    RGB_Map[1, :, :] = heightMap[:cnf.BEV_HEIGHT, :cnf.BEV_WIDTH]  # g_map
    RGB_Map[0, :, :] = intensityMap[:cnf.BEV_HEIGHT, :cnf.BEV_WIDTH]  # b_map
    return RGB_Map


# bev image coordinates format  计算bbox的四个角的坐标
def get_corners(x, y, w, l, yaw):
    bev_corners = np.zeros((4, 2), dtype=np.float32)
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    # front left
    bev_corners[0, 0] = x - w / 2 * cos_yaw - l / 2 * sin_yaw
    bev_corners[0, 1] = y - w / 2 * sin_yaw + l / 2 * cos_yaw

    # rear left
    bev_corners[1, 0] = x - w / 2 * cos_yaw + l / 2 * sin_yaw
    bev_corners[1, 1] = y - w / 2 * sin_yaw - l / 2 * cos_yaw

    # rear right
    bev_corners[2, 0] = x + w / 2 * cos_yaw + l / 2 * sin_yaw
    bev_corners[2, 1] = y + w / 2 * sin_yaw - l / 2 * cos_yaw

    # front right
    bev_corners[3, 0] = x + w / 2 * cos_yaw - l / 2 * sin_yaw
    bev_corners[3, 1] = y + w / 2 * sin_yaw + l / 2 * cos_yaw

    return bev_corners


def drawRotatedBox(img, x, y, w, l, yaw, color):
    # 计算bbox的四个角的坐标
    bev_corners = get_corners(x, y, w, l, yaw)
    corners_int = bev_corners.reshape(-1, 1, 2).astype(int)
    # 用红色画出边框
    cv2.polylines(img, [corners_int], True, color, 2) # True表示是否闭合 isclosed属性
    corners_int = bev_corners.reshape(-1, 2)
    # 把朝向的那条边专门标注出来
    cv2.line(img, (int(corners_int[0, 0]), int(corners_int[0, 1])), (int(corners_int[3, 0]), int(corners_int[3, 1])), (255, 255, 0), 2)
    # cv2.imshow("img",img)