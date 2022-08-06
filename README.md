# Targetless Extrinsic Calibration of Stereo, Thermal, and Laser Sensors in Structured Environments

A new targetless cross-modal calibration system focusing on the extrinsic transformations among stereo cameras, thermal cameras, and laser sensors. Our method requires no dedicated targets and performs the multi-sensor calibration in a single shot without human interaction.

![image](https://github.com/FuTaimeng/auto_calibration/blob/main/figures/laser_calib.png)
Laser points projected on RGB image (converted to grayscale for better visualization) with initial and calibrated extrinsic.

![image](https://github.com/FuTaimeng/auto_calibration/blob/main/figures/th_calib_4.png)
Stereo and laser edge projections (red dots) on thermal images before and after thermal extrinsic calibration on four datasets.

## Enviroments
Please use the DockerFile we provided to configure the enviroment.

## Installation
```
mkdir -p ~/calib_ws/src
cd ~/calib_ws/src
git clone https://github.com/FuTaimeng/auto_calib.git
cd ..
catkin_make
source devel/setup.bash
```
## Run Quick Test
### Download test data
[scene1.tar.xz](https://cuhko365-my.sharepoint.com/:u:/g/personal/119010073_link_cuhk_edu_cn/Efkif2D7whpEqsHae2zKc_4BLdxVhcL_TSyGuGX8-UMs7Q?e=WdK8fg)
### Laser to stereo calibration
```
roslaunch auto_calibration lidar_extrinsic_optimizer_node.launch path:=/path/to/dataset
```
### Stereo-and-laser to thermal calibration
```
roslaunch auto_calibration thermal_extrinsic_optimizer_node.launch path:=/path/to/dataset
```
### Stereo-only to laser calibration
```
roslaunch auto_calibration thermal_extrinsic_optimizer_node.launch path:=/path/to/dataset cloud_dir:=stereo_clouds_edge
```

## Run from begining
### Download bag files
[camera_100.bag](https://drive.google.com/file/d/1syumNfvM3CZVBUttvkKv0DTnUdZwwFrM/view?usp=sharing)

[lidar_100.bag](https://drive.google.com/file/d/1igsIG6DP7nR5VegzC9-9-CkyTS5x_WXC/view?usp=sharing)
### Extract data
```
roslaunch auto_calibration extraction_node.launch path:=/path/to/dataset & \
rosbag play -d 1 -r 0.5 path/to/bagfile/camera_100.bag & \
rosbag play -d 1 -r 0.5 path/to/bagfile/lidar_100.bag
```
### Undistort images
Prepare stereo and thermal intrinsic calibration files in `calib/stereo` and `calib/thermal`. You can use the files we provided.
```
roslaunch auto_calibration image_processor_node.launch path:=/path/to/dataset
```
### Stereo matching and triangulation
```
roslaunch auto_calibration stereo_pair_feature_matcher_node.launch path:=/path/to/dataset
roslaunch auto_calibration stereo_triangulation_node.launch path:=/path/to/dataset
```
### Laser to stereo calibration
```
roslaunch auto_calibration lidar_extrinsic_optimizer_node.launch path:=/path/to/dataset
```
### Stereo edge cloud generation
```
python ~/calib_ws/src/auto_calibration/src/stereo_edge_flow.py /path/to/dataset
```
### Stereo-only to laser calibration
```
roslaunch auto_calibration thermal_extrinsic_optimizer_node.launch path:=/path/to/dataset cloud_dir:=stereo_clouds_edge
```
### Merge stereo and laser clouds
```
roslaunch auto_calibration merge_lidar_stereo_cloud_node.launch path:=/path/to/dataset
```
### Stereo-and-laser to thermal calibration
```
roslaunch auto_calibration thermal_extrinsic_optimizer_node.launch path:=/path/to/dataset
```

## Cite
```
@article{fu2021targetless,
  title={Targetless Extrinsic Calibration of Stereo Cameras, Thermal Cameras, and Laser Sensors in the Wild},
  author={Fu, Taimeng and Yu, Huai and Hu, Yaoyu and Scherer, Sebastian},
  journal={arXiv preprint arXiv:2109.13414},
  year={2021}
}
```
