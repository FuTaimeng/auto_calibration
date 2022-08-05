#ifndef __MODEL_IO_HPP__
#define __MODEL_IO_HPP__

#include <fstream>
#include <vector>

#include <Eigen/Core>

#include <opencv2/core.hpp>

#include <boost/filesystem.hpp>
#include <boost/function.hpp>

#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>

void outputPointsToPLY(const std::vector<cv::Point3d> &points, const std::string filename);
void outputPointsToPLY(const pcl::PointCloud<pcl::PointXYZRGB> &cloud, const std::string filename);

void outputPointsToPLY(const std::vector<cv::Point3d> &points, const std::string filename) {
    pcl::PointCloud<pcl::PointXYZ> cloud;
    for(auto &pt : points) {
        pcl::PointXYZ p(pt.x, pt.y, pt.z);
        cloud.push_back(p);
    }
    pcl::PLYWriter writer;
    writer.write(filename, cloud, true, false);
}

void outputPointsToPLY(const std::vector<cv::Point3d> &points, 
                       const std::vector<cv::Scalar> &colors, 
                       const std::string filename) {
    pcl::PointCloud<pcl::PointXYZRGB> cloud;
    int num_points = points.size();
    for (int i=0; i<num_points; i++) {
        pcl::PointXYZRGB p;
        p.x = points[i].x, p.y = points[i].y, p.z = points[i].z;
        p.r = colors[i].val[2], p.g = colors[i].val[1], p.b = colors[i].val[0];
        cloud.push_back(p);
    }
    pcl::PLYWriter writer;
    writer.write(filename, cloud, true, false);
}

#endif
