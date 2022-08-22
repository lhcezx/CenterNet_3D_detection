import os
import argparse
from easydict import EasyDict as edict
import math
import pdb


def param(arg = ''):
    parser = argparse.ArgumentParser(description='dataset_spliter')
    parser.add_argument('--data_dir', default='../../dataset/kitti')
    parser.add_argument('--split', default='training', help = "which dataset need to be splited")
    parser.add_argument('--image_path',default = 'image_2')
    parser.add_argument('--velodyne_path',default = 'velodyne')
    parser.add_argument('--calibration_path',default = 'calib')
    parser.add_argument('--label_path',default = 'label_2')

    if arg == '':
        param = parser.parse_args()
        # param = edict(vars(parser.parse_args()))  # vars返回对象的属性和属性值的字典， EasyDict让我们像访问属性一样访问dict里的变量
    else:
        param = parser.parse_args()
    
    param.image_path = os.path.join(param.data_dir,param.split,param.image_path)
    param.velodyne_path = os.path.join(param.data_dir,param.split,param.velodyne_path)
    param.calibration_path = os.path.join(param.data_dir,param.split,param.calibration_path)
    param.label_path = os.path.join(param.data_dir,param.split,param.label_path)
    param.total = len(os.listdir(param.image_path))
    param.train_num = math.ceil(param.total*0.7)
    return param


        
class dataset_split(object):
    def __init__(self,param):
        self.param = param 
    
    # 写txt文件
    def write(self, spliter = "", index = 0, mode = "train"):
        spliter = self.spliter()
        index = self.param.train_num if mode == "val" else 0
        with open('{}.txt'.format(mode),"w") as f: 
            _ = self.param.train_num if mode == "train" else self.param.total 
            while index < _:
                f.write(spliter["img"][index]+"\n")
                index+=1

    # 返回一个字典
    def spliter(self):
        assert self.param.data_dir, "data_path error"
        ret = {}
        ret["img"] = os.listdir(self.param.image_path)
        return ret



def main():
    dataset = dataset_split(param())

    dataset.write()
    dataset.write(mode = "val")
    
if __name__ == '__main__':
    main()

