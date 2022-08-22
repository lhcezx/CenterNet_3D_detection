# CenterNet_3D_detection
This project is for 3D object detection, which combines CenterNet, Feature Pyramid Networks and PointPillars. 

# Data preparation
If you use KITTI dataset, here is the [link](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d).
You need to download the data declared below:

- Velodyne point clouds (29 GB)
- Training labels of object data set (5 MB)
- Camera calibration matrices of object data set (16 MB)
- Left color images of object data set (12 GB) (For visualization purpose only)

# Getting Started

### Demo
```
python demo_2_sides.py
```

### Train with PointPillars
In our case we got better results using PointPillars, PointPillars converts point clouds into voxels, and finally converts voxels into Pillars. A PP network is added in front of the backbone network. Different from the original work, we use Anchor Free's CenterNet and detection network with FPN.
```
python train_pp.py 
```

### Train without PointPillars
```
python train.py 
```

### Test
In order to test the model results, you first need to decode the prediction results and save them in KITTI format. To do that, you need to run
```
python test.py 
```

Then you need to use C++ in cpp folder to perform model inference on the saved results.
It should be noted that if you use the KITTI dataset, the test set won't have labels, so it is recommended to use the validation set for testing
```
g++ -O3 -DNDEBUG -o evaluate_object evaluate_object.cpp
```
Eventually you should be able to draw the PR curve for 2D BEV BBox and 3D BBox, you also can use this files to calculate the map.

![image](https://user-images.githubusercontent.com/81321338/185967405-bb3b9d9e-46b6-41f1-ad92-080e91dea709.png)
![image](https://user-images.githubusercontent.com/81321338/185967730-6d3c9c0e-9cbc-4bd1-91f8-db46747f2201.png)


## Folder structure
```
${ROOT}
└── checkpoints/
    ├── fpn_resnet_18/    
        ├── fpn_resnet_18_epoch_300.pth
└── dataset/    
    └── kitti/
        ├──ImageSets/
        │   ├── test.txt
        │   ├── train.txt
        │   └── val.txt
        ├── training/
        │   ├── image_2/ (left color camera)
        │   ├── calib/
        │   ├── label_2/
        │   └── velodyne/
        └── testing/  
        │   ├── image_2/ (left color camera)
        │   ├── calib/
        │   └── velodyne/
        └── classes_names.txt
└── sfa/
├── README.md 
└── requirements.txt
```


## References

[1] CenterNet: [Objects as Points paper](https://arxiv.org/abs/1904.07850), [PyTorch Implementation](https://github.com/xingyizhou/CenterNet) <br>
[2] PointPillars: [Fast Encoders for Object Detection from Point Clouds](https://arxiv.org/abs/1812.05784), [PyTorch Implementation](https://github.com/nutonomy/second.pytorch) <br>
[3] Super Fast and Accurate 3D Object Detection: [PyTorch implementation](https://github.com/maudzung/SFA3D) <br>
[4] [Feature Pyramid Networks for Object Detection](https://arxiv.org/abs/1612.03144) <br>
