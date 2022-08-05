#include <ros/ros.h>

#include "Eigen/Core"

#include "model_io.hpp"
#include "file_io.hpp"
#include "pose3d.hpp"


typedef pcl::PointXYZ PointT;
typedef pcl::PointCloud<PointT> PointCloudT;

void transformPointCloud( const PointCloudT &src,
                          PointCloudT &res,
                          const Mat3 &R, const Vec3 &C );


int main(int argc, char** argv) {
    ros::init(argc, argv, "merge_lidar_stereo_cloud_node");
    ros::NodeHandle n("~");

    std::string lidarCalibDir, lidarCloudDir, stereoCloudDir, mergedCloudDir, targetType;
    n.param("lidar_calib_dir",      lidarCalibDir,          std::string(""));
    n.param("lidar_cloud_dir",      lidarCloudDir,          std::string(""));
    n.param("stereo_cloud_dir",     stereoCloudDir,         std::string(""));
    n.param("merged_cloud_dir",     mergedCloudDir,         std::string(""));
    n.param("target_type",          targetType,             std::string("sl"));

    createCleanFolder(mergedCloudDir);

    omp_lock_t writelock;
    omp_init_lock(&writelock);

    // ------------------------------------
    // 1. load lidar point clouds
    // ------------------------------------
    std::vector<std::string> lidarCloudFilenames;
    readData(lidarCloudDir+"/cloud_filelist.csv", lidarCloudFilenames);

    int nCloud = lidarCloudFilenames.size();

    Mat R, C;
    readData(lidarCalibDir+"/R.dat", 3, 3, ' ', R);
    readData(lidarCalibDir+"/T.dat", 3, 1, ' ', C);

    std::vector<PointCloudT> lidarClouds(nCloud);

    int count = 0;
    #pragma omp parallel for
    for (int i=0; i<nCloud; i++) {
        PointCloudT cloud;
        if (pcl::io::loadPLYFile(lidarCloudFilenames[i], cloud) == -1) {
            std::cerr << "Failed to read: " << lidarCloudFilenames[i] << std::endl;
            continue;
        }

        PointCloudT regCloud;
        transformPointCloud(cloud, regCloud, R, C);

        omp_set_lock(&writelock);
        std::cout << "Load lidar cloud " << ++count << "/" << nCloud << "\r" << std::flush;
        lidarClouds[i] = std::move(regCloud);
        omp_unset_lock(&writelock);
    }
    std::cout << std::endl;
    std::cout << "Load lidar clouds done." << std::endl;

    // ------------------------------------
    // 2. load stereo point clouds
    // ------------------------------------
    std::vector<PointCloudT> stereoClouds(nCloud);

    count = 0;
    #pragma omp parallel for
    for (int i=0; i<nCloud; i++) {
        PointCloudT cloud;
        int p1 = lidarCloudFilenames[i].find_last_of('/');
        int p2 = lidarCloudFilenames[i].find_last_of('.');
        std::string idxStr = lidarCloudFilenames[i].substr(p1+1, p2-p1-1);
        std::string fname = stereoCloudDir+"/"+idxStr+".ply";
        if (pcl::io::loadPLYFile(fname, cloud) == -1) {
            std::cerr << "Failed to read: " << fname << std::endl;
            continue;
        }

        omp_set_lock(&writelock);
        std::cout << "Load stereo cloud " << ++count << "/" << nCloud << "\r" << std::flush;
        stereoClouds[i] = std::move(cloud);
        omp_unset_lock(&writelock);
    }
    std::cout << std::endl;
    std::cout << "Load stereo clouds done." << std::endl;

    // ------------------------------------
    // 3. merge clouds
    // ------------------------------------
    std::vector<PointCloudT> mergedClouds(nCloud);

    count = 0;
    #pragma omp parallel for
    for (int i=0; i<nCloud; i++) {
        PointCloudT cloud;
        if (targetType.find("s") != -1) {
            cloud += stereoClouds[i];
        }
        if (targetType.find("l") != -1) {
            cloud += lidarClouds[i];
        }

        int p1 = lidarCloudFilenames[i].find_last_of('/');
        int p2 = lidarCloudFilenames[i].find_last_of('.');
        std::string idxStr = lidarCloudFilenames[i].substr(p1+1, p2-p1-1);
        std::string fname = mergedCloudDir+"/"+idxStr+".ply";
        pcl::PLYWriter writer;
        writer.write(fname, cloud, false, false);

        omp_set_lock(&writelock);
        std::cout << "Merged cloud " << ++count << "/" << nCloud << "\r" << std::flush;
        omp_unset_lock(&writelock);
    }
    std::cout << std::endl;
    std::cout << "Merge clouds done." << std::endl;

    return 0;
}

void transformPointCloud( const PointCloudT &src,
                          PointCloudT &res,
                          const Mat3 &R, const Vec3 &C ) {
    
    for (const auto &pt : src.points) {
        pcl::PointXYZ np;
        Vec3 X;
        X << pt.x, pt.y, pt.z;
        X = R * X + C;
        np.x = X(0); np.y = X(1); np.z = X(2);
        res.push_back(np);
    }
}
