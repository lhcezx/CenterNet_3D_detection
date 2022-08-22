import os
import sys
import numpy as np

src_dir = os.path.dirname(os.path.realpath(__file__))
while not src_dir.endswith("sfa"):
    src_dir = os.path.dirname(src_dir) # 目录路径循环直到sfa为src_dir
if src_dir not in sys.path:
    sys.path.append(src_dir)


def id_inverse(num):
    if num == 1:
        id_str = "Car"
    elif num == 0:
        id_str = "Pedestrian"
    elif num == 2:
        id_str = "Cyclist"
    return id_str


def writer(f, kitti_dets):
    box_coord = "0.00 0.00 0.00 0.00"
    trun = "0.00"
    occlu = "0"
    alpha = "-10.00"
    for j in range(kitti_dets.shape[0]):
        cls = id_inverse(kitti_dets[j][0])  
        center_3d = [(str(round(k,2))+" ") for k in kitti_dets[j][1:4]]
        dim = [(str(round(k,2))+" ") for k in kitti_dets[j][4:7]]
        ry = round(kitti_dets[j][7],2)
        score = round(kitti_dets[j][8],2)
        f.write("{} {} {} {} {} ".format(cls, trun, occlu, alpha, box_coord))
        f.writelines(dim)
        f.writelines(center_3d)
        f.write("{} {}\n".format(ry, score))


def detection_eval(img_path, kitti_dets, output_path):
    split_txt = img_path.replace("png","txt").split("/")[-1]
    output_path = os.path.join(output_path, split_txt)
    with open(output_path,"a+") as f:
        writer(f, kitti_dets)

def clean(path):
    if os.path.isdir(path) and os.listdir(path): # 如果文件夹存在且非空，删除内部所有文件
        for file in os.listdir(path):
            os.remove(path + "/" + file)


if __name__ == "__main__":
    det = np.array([0.1111111111,0.2222222,0.3333333,0.44444444444,0.555555555,0.6666666666,0.7777777777,0.88888888888]).reshape(1,8)
    with open("test.txt","a+") as f:
        writer(f, det)