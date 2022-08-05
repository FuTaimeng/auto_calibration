DATA_PATH="/home/tymon/DataSSD/shimizu/scene1"
BAG_PATH="/home/tymon/DataSSD/shimizu/bags"

roslaunch auto_calibration extraction_node.launch \
          path:=${DATA_PATH} & \
rosbag play -d 1 -r 0.5 ${BAG_PATH}/camera_100.bag & \
rosbag play -d 1 -r 0.5 ${BAG_PATH}/lidar_100.bag
