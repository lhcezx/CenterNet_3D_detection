import sys
import os
import math
from builtins import int
import numpy as np
from torch.utils.data import Dataset
import cv2
import torch

# 返回当前脚本的绝对路径
src_dir = os.path.dirname(os.path.realpath(__file__))
while not src_dir.endswith("sfa"):
    src_dir = os.path.dirname(src_dir) # 目录路径循环直到sfa为src_dir
if src_dir not in sys.path:
    sys.path.append(src_dir)

from data_process.kitti_data_utils import gen_hm_radius, compute_radius, Calibration, get_filtered_lidar
from data_process.kitti_bev_utils import makeBEVMap, drawRotatedBox, get_corners
from data_process import transformation
import config.kitti_config as cnf


class KittiDataset(Dataset):
    def __init__(self, configs, mode='train', lidar_aug=None, hflip_prob=None, num_samples=None):
        self.dataset_dir = configs.dataset_dir  # dataset/kitti
        self.input_size = configs.input_size
        self.hm_size = configs.hm_size

        self.num_classes = configs.num_classes
        self.max_objects = configs.max_objects

        assert mode in ['train','test','trainval',"val"], 'Invalid mode: {}'.format(mode)
        self.mode = mode
        self.is_test = (self.mode == 'test')
        # sub_folder = 'testing' if self.is_test else 'training'
        sub_folder = 'training'

        self.lidar_aug = lidar_aug
        self.hflip_prob = hflip_prob

        self.image_dir = os.path.join(self.dataset_dir, sub_folder, "image_2")  # 图像数据.png
        self.lidar_dir = os.path.join(self.dataset_dir, sub_folder, "velodyne") # 激光雷达数据.bin
        self.calib_dir = os.path.join(self.dataset_dir, sub_folder, "calib")  # 相机标定文件
        self.label_dir = os.path.join(self.dataset_dir, sub_folder, "label_2") # 标签文件
        split_txt_path = os.path.join(self.dataset_dir, 'ImageSets', 'val.txt').replace("\\","/") if self.is_test else os.path.join(self.dataset_dir, 'ImageSets', '{}.txt'.format(mode)).replace("\\","/") # train, val_dev, val, test其中之一的txt
        self.sample_id_list = [int(x.strip()) for x in open(split_txt_path).readlines()]  #.strip() 用于移除字符串头尾指定的字符（默认为空格或换行符）或字符序列


        if num_samples is not None:
            self.sample_id_list = self.sample_id_list[:num_samples]  # 如果限定样本数量，则截断num_sample后面的数据
        self.num_samples = len(self.sample_id_list) 

    def __len__(self):
        return len(self.sample_id_list)

    def __getitem__(self, index):
        if self.is_test:
            return self.load_img_only(index)
        else:
            return self.load_img_with_targets(index)

    def load_img_only(self, index):
        """Load only image for the testing phase"""
        sample_id = int(self.sample_id_list[index])
        img_path= self.get_image(sample_id)
        lidarData = self.get_lidar(sample_id)
        lidarData = get_filtered_lidar(lidarData, cnf.boundary)  # 过滤雷达信息
        bev_map = makeBEVMap(lidarData, cnf.boundary)  # 通过点云数据生成鸟瞰图BEV
        bev_map = torch.from_numpy(bev_map)

        metadatas = {
            'img_path': img_path,
        }
        return metadatas, bev_map

    # 读取图片并build_targets，对数据集进行丰富，通常用于训练
    def load_img_with_targets(self, index):
        """Load images and targets for the training and validation phase"""
        sample_id = int(self.sample_id_list[index])
        img_path = os.path.join(self.image_dir, '{:06d}.png'.format(sample_id))
        lidarData = self.get_lidar(sample_id)
        calib = self.get_calib(sample_id)
        labels, has_labels = self.get_label(sample_id)  # 读取label并且忽略 id<=-99(tram,misc)的label.
        if has_labels:
            labels[:, 1:] = transformation.camera_to_lidar_box(labels[:, 1:], calib.V2C, calib.R0, calib.P2)

        # lidar_aug 可以是一个定义在transformation里面的Oneof类，重载了__call__函数，可以把这个类当函数来使用用于增强雷达的数据
        if self.lidar_aug:
            lidarData, labels[:, 1:] = self.lidar_aug(lidarData, labels[:, 1:]) # lidar_aug是一个初始化的Oneof实例，将随机选择一种方式进行数据增强

        # 过滤点云信息
        lidarData, labels = get_filtered_lidar(lidarData, cnf.boundary, labels)
        # 生成鸟瞰图并将其转为tensor
        bev_map = makeBEVMap(lidarData, cnf.boundary)
        bev_map = torch.from_numpy(bev_map)

        # 是否进行随机翻转
        hflipped = False
        if np.random.random() < self.hflip_prob:
            hflipped = True
            # C, H, W
            bev_map = torch.flip(bev_map, [-1]) # 左右翻转

        targets = self.build_targets(labels, hflipped)

        metadatas = {
            'img_path': img_path,
            'hflipped': hflipped
        }

        return metadatas, bev_map, targets

    # 返回图片路径和RGB图像
    def get_image(self, idx):
        img_path = os.path.join(self.image_dir, '{:06d}.png'.format(idx))
        # img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) # BGR转换成RGB

        return img_path

    # 通过calib_file字典实例化一个Calibration类
    def get_calib(self, idx):
        calib_file = os.path.join(self.calib_dir, '{:06d}.txt'.format(idx))
        # assert os.path.isfile(calib_file)
        return Calibration(calib_file) 

    #读取lidar的信息
    def get_lidar(self, idx):
        lidar_file = os.path.join(self.lidar_dir, '{:06d}.bin'.format(idx))
        # assert os.path.isfile(lidar_file)
        return np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 4) # 从lidar_file文件中读取数据并reshape成(N, 4),点云数据pointcloud包括(x,y,z，反射强度)

    # 返回一个标签列表, 其中包含(cat_id,x,y,z,h,w,l,ry),dim = 8
    def get_label(self, idx):
        labels = []
        label_path = os.path.join(self.label_dir, '{:06d}.txt'.format(idx))
        for line in open(label_path, 'r'):
            line = line.rstrip()
            line_parts = line.split(' ')
            obj_name = line_parts[0]  # 'Car', 'Pedestrian', ...
            cat_id = int(cnf.CLASS_NAME_TO_ID[obj_name])
            if cat_id <= -99:  # ignore Tram and Misc
                continue
            truncated = int(float(line_parts[1]))  # truncated pixel ratio [0..1]  是否物体在图片外被截断，截断程度
            occluded = int(line_parts[2])  # 0=visible, 1=partly occluded, 2=fully occluded, 3=unknown  # 遮挡率
            alpha = float(line_parts[3])  # object observation angle [-pi..pi]   # 观察角度
            # xmin, ymin, xmax, ymax 
            bbox = np.array([float(line_parts[4]), float(line_parts[5]), float(line_parts[6]), float(line_parts[7])]) # 2D bbox左上和右下点坐标x,y
            # height, width, length (h, w, l)
            h, w, l = float(line_parts[8]), float(line_parts[9]), float(line_parts[10]) # 三维物体的长宽高
            # location (x,y,z) in camera coord.
            x, y, z = float(line_parts[11]), float(line_parts[12]), float(line_parts[13]) # 三维物体的在照相机坐标系下的坐标，单位m
            ry = float(line_parts[14])  # yaw angle (around Y-axis in camera coordinates) [-pi..pi] # 偏航角，在照相机坐标系下，物体的全局方向角（物体前进方向与相机坐标系x轴的夹角）

            object_label = [cat_id, x, y, z, h, w, l, ry]
            labels.append(object_label)

        if len(labels) == 0:
            labels = np.zeros((1, 8), dtype=np.float32)
            has_labels = False
        else:
            labels = np.array(labels, dtype=np.float32)
            has_labels = True

        return labels, has_labels

    # 返回一个字典，字典里面每一个值包括了所有的GT的信息(hm,off,direction,z_coor，dim,ind,mask)
    def build_targets(self, labels, hflipped):
        minX = cnf.boundary['minX']
        maxX = cnf.boundary['maxX']
        minY = cnf.boundary['minY']
        maxY = cnf.boundary['maxY']
        minZ = cnf.boundary['minZ']
        maxZ = cnf.boundary['maxZ']

        num_objects = min(len(labels), self.max_objects)
        hm_l, hm_w = self.hm_size

        hm_main_center = np.zeros((self.num_classes, hm_l, hm_w), dtype=np.float32) # 每个类的热图hw
        cen_offset = np.zeros((self.max_objects, 2), dtype=np.float32) # 每个物体的中心点偏置
        direction = np.zeros((self.max_objects, 2), dtype=np.float32) # 每个物体的方向
        z_coor = np.zeros((self.max_objects, 1), dtype=np.float32) # 每个物体的Z坐标
        dimension = np.zeros((self.max_objects, 3), dtype=np.float32) # 每个物体的size

        indices_center = np.zeros((self.max_objects), dtype=np.int64) # 下标
        obj_mask = np.zeros((self.max_objects), dtype=np.uint8) # 是否mask回归

        for k in range(num_objects):
            cls_id, x, y, z, h, w, l, yaw = labels[k]
            cls_id = int(cls_id)
            # Invert yaw angle
            yaw = -yaw
            if not ((minX <= x <= maxX) and (minY <= y <= maxY) and (minZ <= z <= maxZ)):
                continue
            if (h <= 0) or (w <= 0) or (l <= 0):
                continue

            bbox_l = l / cnf.bound_size_x * hm_l # 把长宽按照热图的尺寸进行缩放
            bbox_w = w / cnf.bound_size_y * hm_w
            radius = compute_radius((math.ceil(bbox_l), math.ceil(bbox_w))) # 计算高斯核半径
            radius = max(0, int(radius)) 
            center_y = (x - minX) / cnf.bound_size_x * hm_l  # x --> y (invert to 2D image space)
            center_x = (y - minY) / cnf.bound_size_y * hm_w  # y --> x
            center = np.array([center_x, center_y], dtype=np.float32)

            if hflipped:
                center[0] = hm_w - center[0] - 1

            center_int = center.astype(np.int32)
            if cls_id < 0: # Truck: -3, 'DontCare': -1, 'Tram': -99
                ignore_ids = [_ for _ in range(self.num_classes)] if cls_id == - 1 else [- cls_id - 2]  # 如果类是卡车(-3), 那就在car这个类的hm上画热点； 如果是dont care类(-1)：就在0, 1, 2三个类上画热点，label = -99的情况已经在get_label中被忽略了
                # Consider to make mask ignore
                for cls_ig in ignore_ids:
                    gen_hm_radius(hm_main_center[cls_ig], center_int, radius)
                hm_main_center[ignore_ids, center_int[1], center_int[0]] = 0.9999 # 为什么要把(y,x)点的值设为0.999
                continue

            # Generate heatmaps for main center 在中心点放置高斯核
            gen_hm_radius(hm_main_center[cls_id], center, radius)
            # Index of the center 通过中心点x,y坐标计算每个物体的中心点下标计算
            indices_center[k] = center_int[1] * hm_w + center_int[0]

            # targets for center offset 中心点偏置
            cen_offset[k] = center - center_int

            # targets for dimension 
            dimension[k, 0] = h
            dimension[k, 1] = w
            dimension[k, 2] = l

            # targets for direction 方向计算
            direction[k, 0] = math.sin(float(yaw))  # im
            direction[k, 1] = math.cos(float(yaw))  # re
            # im -->> -im
            if hflipped:
                direction[k, 0] = - direction[k, 0]

            # targets for depth 深度计算
            z_coor[k] = z - minZ

            # Generate object masks 
            obj_mask[k] = 1

        targets = {
            'hm_cen': hm_main_center,
            'cen_offset': cen_offset,
            'direction': direction,
            'z_coor': z_coor,
            'dim': dimension,
            'indices_center': indices_center,
            'obj_mask': obj_mask,
        }
        
        return targets
    
    # 用于画出图片和标签，用于demo
    def draw_img_with_label(self, index):
        sample_id = int(self.sample_id_list[index])
        img_path, img_rgb = self.get_image(sample_id)
        lidarData = self.get_lidar(sample_id)
        calib = self.get_calib(sample_id)
        labels, has_labels = self.get_label(sample_id)
        if has_labels:
            # 把相机坐标系下的label数据转到激光雷达坐标系下
            labels[:, 1:] = transformation.camera_to_lidar_box(labels[:, 1:], calib.V2C, calib.R0, calib.P2)
        # 是否进行数据增强
        if self.lidar_aug:
            lidarData, labels[:, 1:] = self.lidar_aug(lidarData, labels[:, 1:]) # lidar_aug是一个初始化的Oneof实例，将随机选择一种方式进行数据增强

        # 过滤数据和标签, 要满足min_max_xyz的图片
        lidarData, labels = get_filtered_lidar(lidarData, cnf.boundary, labels)
        bev_map = makeBEVMap(lidarData, cnf.boundary)

        return bev_map, labels, img_rgb, img_path


if __name__ == '__main__':
    from easydict import EasyDict as edict
    from data_process.transformation import OneOf, Random_Scaling, Random_Rotation, lidar_to_camera_box
    from utils.visualization_utils import merge_rgb_to_bev, show_rgb_image_with_boxes

    configs = edict() # 可以使得以属性的方式去访问字典的值
    configs.distributed = False  # For testing
    configs.pin_memory = False
    configs.num_samples = None
    configs.input_size = (608, 608)
    configs.hm_size = (152, 152)
    configs.max_objects = 50
    configs.num_classes = 3 
    configs.output_width = 608

    configs.dataset_dir = os.path.join('../../', 'dataset', 'kitti')
    # lidar_aug = OneOf([
    #     Random_Rotation(limit_angle=np.pi / 4, p=1.),
    #     Random_Scaling(scaling_range=(0.95, 1.05), p=1.),
    # ], p=1.)
    lidar_aug = None
    
    # 创建数据集
    dataset = KittiDataset(configs, mode='val', lidar_aug=lidar_aug, hflip_prob=0., num_samples=configs.num_samples)

    print('\n\nPress n to see the next sample >>> Press Esc to quit...')
    # 每一个数据画出鸟瞰图, label, RGB图像和其路径
    for idx in range(len(dataset)):
        bev_map, labels, img_rgb, img_path = dataset.draw_img_with_label(idx)
        # 通过calib文件路径创建一个校准类
        calib = Calibration(img_path.replace(".png", ".txt").replace("image_2", "calib"))
        # CHW 2 HWC
        bev_map = (bev_map.transpose(1, 2, 0) * 255).astype(np.uint8)
        
        bev_map = cv2.resize(bev_map, (cnf.BEV_HEIGHT, cnf.BEV_WIDTH))
        # 读取标签信息并在鸟瞰图上画出bbox
        for box_idx, (cls_id, x, y, z, h, w, l, yaw) in enumerate(labels):
            # Draw rotated box
            yaw = -yaw
            y1 = int((x - cnf.boundary['minX']) / cnf.DISCRETIZATION)
            x1 = int((y - cnf.boundary['minY']) / cnf.DISCRETIZATION)
            w1 = int(w / cnf.DISCRETIZATION)
            l1 = int(l / cnf.DISCRETIZATION)
            drawRotatedBox(bev_map, x1, y1, w1, l1, yaw, cnf.colors[int(cls_id)])
        # Rotate the bev_map 旋转图像180度
        bev_map = cv2.rotate(bev_map, cv2.ROTATE_180)
        #  x, y, z, h, w, l, rz，激光雷达的信息转换成图像的label
        labels[:, 1:] = lidar_to_camera_box(labels[:, 1:], calib.V2C, calib.R0, calib.P2)
        img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        img_rgb = show_rgb_image_with_boxes(img_rgb, labels, calib) # 把3d的图像信息投影到2d图像，在RGB图像上画出BBOX
        out_img = merge_rgb_to_bev(img_rgb, bev_map, output_width=configs.output_width) # BEV和RGB结合
        cv2.imshow('bev_map', out_img)

        if cv2.waitKey(0) & 0xff == 27:
            break
