cmake_minimum_required(VERSION 2.8.3)
project(auto_calibration)

# Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++11)


find_package(catkin REQUIRED COMPONENTS
    roscpp
    rospy
    std_msgs
    cv_bridge
)

find_package(Eigen3 REQUIRED) 
include_directories(${Eigen3_INCLUDE_DIRS})

find_package(Ceres 1.14 REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})

find_package(OpenCV 3.4.3 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(PCL 1.8 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})

find_package(Boost REQUIRED COMPONENTS filesystem)
include_directories(${Boost_INCLUDE_DIRS})

find_package(OpenMVG REQUIRED)
include_directories(${OPENMVG_INCLUDE_DIRS})

find_package(OpenMP REQUIRED)
if (OPENMP_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}   ${OpenMP_C_FLAGS}")
endif()

catkin_package(
) 
    
include_directories(
    ${catkin_INCLUDE_DIRS}
)

add_executable(stereo_pair_feature_matcher_node src/stereo_pair_feature_matcher.cpp)
target_link_libraries(stereo_pair_feature_matcher_node 
    ${catkin_LIBRARIES} 
    ${OpenCV_LIBRARIES} 
    ${OPENMVG_LIBRARIES} 
    OpenMVG::openMVG_sfm
    OpenMVG::openMVG_camera
    OpenMVG::openMVG_features
    OpenMVG::openMVG_geometry
)

add_executable(stereo_triangulation_node src/stereo_triangulation.cpp)
target_link_libraries(stereo_triangulation_node 
    ${catkin_LIBRARIES} 
    ${OpenCV_LIBRARIES} 
    ${PCL_LIBRARIES} 
    ${OPENMVG_LIBRARIES} 
    OpenMVG::openMVG_sfm
    OpenMVG::openMVG_camera
    OpenMVG::openMVG_features
    OpenMVG::openMVG_geometry
)

add_executable(thermal_extrinsic_optimizer_node src/thermal_extrinsic_optimizer.cpp)
target_link_libraries(thermal_extrinsic_optimizer_node 
    ${catkin_LIBRARIES} 
    ${OpenCV_LIBRARIES} 
    ${PCL_LIBRARIES} 
    ${OPENMVG_LIBRARIES} 
    OpenMVG::openMVG_sfm
    OpenMVG::openMVG_camera
    OpenMVG::openMVG_features
    OpenMVG::openMVG_geometry
)

add_executable(lidar_extrinsic_optimizer_node src/lidar_extrinsic_optimizer.cpp)
target_link_libraries(lidar_extrinsic_optimizer_node 
    ${catkin_LIBRARIES} 
    ${OpenCV_LIBRARIES} 
    ${PCL_LIBRARIES} 
    ${OPENMVG_LIBRARIES} 
    OpenMVG::openMVG_sfm
    OpenMVG::openMVG_camera
    OpenMVG::openMVG_features
    OpenMVG::openMVG_geometry
)

add_executable(extraction_node src/extraction.cpp)
target_link_libraries(extraction_node 
    ${catkin_LIBRARIES} 
    ${OpenCV_LIBRARIES} 
    ${PCL_LIBRARIES} 
)

add_executable(image_preprocess_node src/image_preprocess.cpp)
target_link_libraries(image_preprocess_node 
    ${catkin_LIBRARIES} 
    ${OpenCV_LIBRARIES} 
)

add_executable(merge_lidar_stereo_cloud_node src/merge_lidar_stereo_cloud.cpp)
target_link_libraries(merge_lidar_stereo_cloud_node 
    ${catkin_LIBRARIES} 
    ${PCL_LIBRARIES} 
)

