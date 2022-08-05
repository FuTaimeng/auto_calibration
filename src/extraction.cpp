#include <mutex>
#include <queue>
#include <thread>
#include <vector>

#include <ros/ros.h>

#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <cv_bridge/cv_bridge.h>

#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/image_encodings.h>

#include <pcl/pcl_base.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/filter_indices.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/statistical_outlier_removal.h>

#include "file_io.hpp"

#define USE_CLOUD_FILTER

std::queue<sensor_msgs::ImageConstPtr> rgb_left_buf;
std::queue<sensor_msgs::ImageConstPtr> rgb_right_buf;
std::queue<sensor_msgs::ImageConstPtr> thermal_buf;
std::queue<sensor_msgs::PointCloud2ConstPtr> laser_buf;

std::mutex m_buf;

std::string rgbDir, thermalDir, laserDir;

double IR_max_intensity, IR_min_intensity;
double laser_max_range, laser_min_range, laser_edge_th;
int laser_edge_sample_radius;
bool debug_output;
int sample_step;

bool isEdgePoint( int i, int width, int height, 
                  const std::vector<double> &dist );

pcl::PointXYZ biasedEdgePoint(int i, int width, int height, 
                              const std::vector<double> &dist,
                              const pcl::PointCloud<pcl::PointXYZ> &cloud);

void thermal_callback(const sensor_msgs::ImageConstPtr &image_msg)
{
    ROS_INFO("thermal received");
    m_buf.lock();
    thermal_buf.push(image_msg);
    m_buf.unlock();
}

void rgb_left_callback(const sensor_msgs::ImageConstPtr &image_msg)
{
    ROS_INFO("rgb left received");
    m_buf.lock();
    rgb_left_buf.push(image_msg);
    m_buf.unlock();
}

void rgb_right_callback(const sensor_msgs::ImageConstPtr &image_msg)
{
    ROS_INFO("rgb right received");
    m_buf.lock();
    rgb_right_buf.push(image_msg);
    m_buf.unlock();
}

void laser_callback(const sensor_msgs::PointCloud2ConstPtr &laser_msg)
{
    ROS_INFO("lidar received");
    m_buf.lock();
    laser_buf.push(laser_msg);
    m_buf.unlock();
}

void process()
{
    unsigned int idx = 0, sample_cnt = 0;
    std::vector<double> IRmins, IRmaxs;

    sensor_msgs::ImageConstPtr rgb_left_msg = NULL;
    sensor_msgs::ImageConstPtr rgb_right_msg = NULL;
    sensor_msgs::ImageConstPtr thermal_msg = NULL;
    sensor_msgs::PointCloud2ConstPtr laser_msg=NULL;
    while (true)
    {
        bool ready = false;

        if (!rgb_left_buf.empty() && !rgb_right_buf.empty())
        {
            auto rgb_time = std::min( rgb_left_buf.front()->header.stamp.toSec(), 
                                      rgb_right_buf.front()->header.stamp.toSec() );
            while (!thermal_buf.empty() && rgb_time > thermal_buf.front()->header.stamp.toSec())
            {
                thermal_msg = thermal_buf.front();
                thermal_buf.pop();
            }
            while (!laser_buf.empty() && rgb_time > laser_buf.front()->header.stamp.toSec())
            {
                laser_msg = laser_buf.front();
                laser_buf.pop();
            }
            
            if (!thermal_buf.empty() && !laser_buf.empty())
            {
                rgb_left_msg = rgb_left_buf.front();
                rgb_left_buf.pop();
                rgb_right_msg = rgb_right_buf.front();
                rgb_right_buf.pop();
                if (thermal_msg == NULL || thermal_buf.front()->header.stamp.toSec() - rgb_time < rgb_time - thermal_msg->header.stamp.toSec())
                {
                    thermal_msg = thermal_buf.front();
                    thermal_buf.pop();
                }
                if (laser_msg == NULL || laser_buf.front()->header.stamp.toSec() - rgb_time < rgb_time - laser_msg->header.stamp.toSec())
                {
                    laser_msg = laser_buf.front();
                    laser_buf.pop();
                }
                ready = true;
            }
        }

        if (ready) {
            if (sample_cnt % sample_step != 0) {
                ready = false;
            }
            sample_cnt ++;
        }

        if(ready)
        {
            std::ofstream ofile;
            pcl::PLYWriter plywriter;

            // thermal
            #define IR16_FORMAT
            #ifdef IR16_FORMAT

            cv_bridge::CvImageConstPtr ptr_thermal;
            ptr_thermal = cv_bridge::toCvCopy(thermal_msg, sensor_msgs::image_encodings::TYPE_16UC1);
            cv::Mat IR16 = ptr_thermal->image;

            double minI, maxI;
            if (IR_min_intensity >= 0) 
            {
                minI = IR_min_intensity;
                maxI = IR_max_intensity;
            }
            else 
            {
                // std::vector<int16_t> vec_data;
                // IR16.reshape(0,1).copyTo(vec_data);
                // int min_idx = vec_data.size()*0.01;
                // int max_idx = vec_data.size()*0.99;
                // std::nth_element(vec_data.begin(), vec_data.begin() + min_idx, vec_data.end());
                // std::nth_element(vec_data.begin(), vec_data.begin() + max_idx, vec_data.end());
                // minI = vec_data[min_idx];
                // maxI = vec_data[max_idx];

                cv::minMaxIdx(IR16, &minI, &maxI);
                
                // int mid_idx = vec_data.size()*0.5;
                // double IR_range = IR_max_intensity;
                // std::nth_element(vec_data.begin(), vec_data.begin() + mid_idx, vec_data.end());
                // minI = vec_data[mid_idx] - IR_range/2;
                // maxI = vec_data[mid_idx] + IR_range/2;
                // std::cout << "minI = " << minI << ", maxI = " << maxI << std::endl;
            }

            IRmins.push_back(minI);
            IRmaxs.push_back(maxI);

            std::string thermal_minmax = thermalDir+"/thermal_minmax.txt";
            ofile.open(thermal_minmax, std::ios::app);
            ofile << minI << " " << maxI << std::endl;
            ofile.close();

            cv::Mat IR8(IR16.rows, IR16.cols, 0);
            for (int i=0; i<IR8.rows*IR8.cols; i++)
            {
                auto pixel = ((uint16_t *)ptr_thermal->image.data)[i];
                pixel = pixel > maxI ? maxI : pixel;
                pixel = pixel < minI ? minI : pixel;
                IR8.data[i] = uchar((pixel - minI) * 255.0 / (maxI - minI));
            }

            #else

            cv_bridge::CvImageConstPtr ptr_thermal;
            ptr_thermal = cv_bridge::toCvCopy(thermal_msg, sensor_msgs::image_encodings::RGB8);
            cv::Mat RGB8 = ptr_thermal->image;

            cv::Mat IR8;
            cv::cvtColor(RGB8, IR8, cv::COLOR_RGB2GRAY);

            #endif

            std::string thermal_file = thermalDir + "/" + std::to_string(idx) + ".png";
            cv::imwrite(thermal_file, IR8);

            std::string thermal_filelist = thermalDir+"/thermal_filelist.csv";
            ofile.open(thermal_filelist, std::ios::app);
            ofile << thermal_file << std::endl;
            ofile.close();

            // rgb left
            cv_bridge::CvImageConstPtr ptr_rgb_left;
            ptr_rgb_left = cv_bridge::toCvCopy(rgb_left_msg, sensor_msgs::image_encodings::BGR8);
            cv::Mat rgb_left = ptr_rgb_left->image;
            std::string rgb_left_file = rgbDir + "/left/" + std::to_string(idx) + ".png";
            cv::imwrite(rgb_left_file, rgb_left);

            // rgb right
            cv_bridge::CvImageConstPtr ptr_rgb_right;
            ptr_rgb_right = cv_bridge::toCvCopy(rgb_right_msg, sensor_msgs::image_encodings::BGR8);
            cv::Mat rgb_right = ptr_rgb_right->image;
            std::string rgb_right_file = rgbDir + "/right/" + std::to_string(idx) + ".png";
            cv::imwrite(rgb_right_file, rgb_right);

            std::string left_filelist = rgbDir+"/left/left_filelist.csv";
            ofile.open(left_filelist, std::ios::app);
            ofile << rgb_left_file << std::endl;
            ofile.close();

            std::string right_filelist = rgbDir+"/right/right_filelist.csv";
            ofile.open(right_filelist, std::ios::app);
            ofile << rgb_right_file << std::endl;
            ofile.close();

            // laser
            pcl::PointCloud<pcl::PointXYZ>::Ptr pointCloud(new pcl::PointCloud<pcl::PointXYZ>());
            pcl::fromROSMsg(*laser_msg, *pointCloud);
#ifdef USE_CLOUD_FILTER

            int nPoint = pointCloud->points.size();
            int width = laser_msg->width, height = laser_msg->height;

            std::vector<double> dist(nPoint);
            for (int i=0; i<nPoint; i++) {
                pcl::PointXYZ p(pointCloud->points[i]);
                dist[i] = std::sqrt(p.x*p.x + p.y*p.y + p.z*p.z);
            }

            std::vector<bool> isInlier(nPoint);
            std::vector<bool> isEdge(nPoint);
            for (int i=0; i<nPoint; i++) 
            {
                if (dist[i] > laser_min_range && dist[i] < laser_max_range) 
                {
                    isInlier[i] = true;
                    isEdge[i] = isEdgePoint(i, width, height, dist);
                }
                else isInlier[i] = false;
            }

            pcl::PointCloud<pcl::PointXYZ> filtered, edges;
            for (int i=0; i<nPoint; i++) 
            {
                if (isInlier[i]) 
                {
                    filtered.push_back(pointCloud->points[i]);
                }
                if (isEdge[i]) {
                    // edges.push_back(biasedEdgePoint(i, width, height, dist, *pointCloud));
                    edges.push_back(pointCloud->points[i]);
                }
            }

            std::string cloud_file = laserDir + "/" + std::to_string(idx) + ".ply";
            plywriter.write(cloud_file, filtered, false, false);

            std::string cloud_filelist = laserDir+"/cloud_filelist.csv";
            ofile.open(cloud_filelist, std::ios::app);
            ofile << cloud_file << std::endl;
            ofile.close();

            std::string edges_file = laserDir + "/edges/" + std::to_string(idx) + ".ply";
            plywriter.write(edges_file, edges, false, false);

            std::string edges_filelist = laserDir+"/edges/cloud_filelist.csv";
            ofile.open(edges_filelist, std::ios::app);
            ofile << edges_file << std::endl;
            ofile.close();

            if (debug_output) {
                pcl::PointCloud<pcl::PointXYZRGB> colored;
                for (int i=0; i<nPoint; i++) 
                {
                    if (!isInlier[i]) continue;
                    pcl::PointXYZ pt(pointCloud->points[i]);
                    pcl::PointXYZRGB p;
                    p.x = pt.x; p.y = pt.y; p.z = pt.z;
                    if (isEdge[i]) {
                        p.r = 255;  p.g = 0;    p.b = 0;
                    }
                    else
                    {
                        p.r = 255;  p.g = 255;  p.b = 255;
                    }
                    colored.push_back(p);
                }

                std::string colored_file = laserDir + "/debug/" + std::to_string(idx) + ".ply";
                plywriter.write(colored_file, colored, false, false);
            }
#else
            std::string cloud_file = laserDir + "/" + std::to_string(idx) + ".ply";
            plywriter.write(cloud_file, *pointCloud, false, false);

            std::string cloud_filelist = laserDir+"/cloud_filelist.csv";
            ofile.open(cloud_filelist, std::ios::app);
            ofile << cloud_file << std::endl;
            ofile.close();
#endif

            idx++;
        }  
    }
}

bool isEdgePoint( int i, int width, int height, 
                  const std::vector<double> &dist ) {

    auto moveLeft = [width] (int i) {
        return i%width!=0 ? i-1 : i+width-1;
    };
    auto moveRight = [width] (int i) {
        return i%width!=width-1 ? i+1 : i-width+1;
    };

    auto check = [&dist] (int i, int j) {
        return dist[j] < 1e-5 || dist[j] - dist[i] > laser_edge_th;
    };
    auto has = [] (const std::vector<bool> &arr, bool b) {
        return std::find(arr.begin(), arr.end(), b) != arr.end();
    };

    std::vector<bool> leftFlag, rightFlag;
    int l=i, r=i;
    for (int k=0; k<laser_edge_sample_radius; k++) {
        l = moveLeft(l);
        leftFlag.push_back(check(i, l));
        r = moveRight(r);
        rightFlag.push_back(check(i, r));
    }

    bool leftFar, leftNear, rightFar, rightNear;
    leftFar = has(leftFlag, true);
    leftNear = has(leftFlag, false);
    rightFar = has(rightFlag, true);
    rightNear = has(rightFlag, false);

    return (leftFar && !leftNear && !rightFar && rightNear) || 
        (!leftFar && leftNear && rightFar && !rightNear);
}

pcl::PointXYZ biasedEdgePoint(int i, int width, int height, 
                              const std::vector<double> &dist,
                              const pcl::PointCloud<pcl::PointXYZ> &cloud) {

    auto moveLeft = [width] (int i) {
        return i%width!=0 ? i-1 : i+width-1;
    };
    auto moveRight = [width] (int i) {
        return i%width!=width-1 ? i+1 : i-width+1;
    };

    auto check = [&dist] (int i, int j) {
        return dist[j] < 1e-5 || dist[j] - dist[i] > laser_edge_th;
    };

    const auto &pts = cloud.points;
    int l=moveLeft(i), r=moveRight(i);
    if (check(i, l) && !check(i, r)) {
        return pcl::PointXYZ( 1.5*pts[i].x - 0.5*pts[r].x,
                              1.5*pts[i].y - 0.5*pts[r].y,
                              1.5*pts[i].z - 0.5*pts[r].z );
    }
    else if (!check(i, l) && check(i, r)) {
        return pcl::PointXYZ( 1.5*pts[i].x - 0.5*pts[l].x,
                              1.5*pts[i].y - 0.5*pts[l].y,
                              1.5*pts[i].z - 0.5*pts[l].z );
    }
    else return pts[i];
}

// bool isEdgePoint( int i, int width, int height, 
//                   const std::vector<double> &dist ) {
//     int nPoint = width * height;
//     auto moveUp = [width, nPoint] (int i) {
//         if (i<0 || i>=nPoint) return -1;
//         else return i-width >= 0 ? i-width : -1;
//     };
//     auto moveDown = [width, nPoint] (int i) {
//         if (i<0 || i>=nPoint) return -1;
//         else return i+width < nPoint ? i+width : -1;
//     };
//     auto moveLeft = [width, nPoint] (int i) {
//         if (i<0 || i>=nPoint) return -1;
//         else return i%width!=0 ? i-1 : i+width-1;
//     };
//     auto moveRight = [width, nPoint] (int i) {
//         if (i<0 || i>=nPoint) return -1;
//         else return i%width!=width-1 ? i+1 : i-width+1;
//     };

//     //       0
//     //    1  2  3
//     // 4  5  i  6  7
//     //    8  9 10
//     //      11

//     std::vector<int> neighbors({
//         moveUp(moveUp(i)),
//         moveLeft(moveUp(i)),
//         moveUp(i),
//         moveRight(moveUp(i)),
//         moveLeft(moveLeft(i)),
//         moveLeft(i),
//         moveRight(i),
//         moveRight(moveRight(i)),
//         moveLeft(moveDown(i)),
//         moveDown(i),
//         moveRight(moveDown(i)),
//         moveDown(moveDown(i))
//     });

//     std::vector<bool> flag(12);
//     for (int k=0; k<12; k++) {
//         if (neighbors[k] != -1) {
//             flag[k] = dist[neighbors[k]] < 1e-5 || 
//                 dist[neighbors[k]] - dist[i] > laser_edge_th;
//         }
//         else flag[k] = true;
//     }
    
//     std::vector<std::vector<int>> checkLists({
//         {0,1,2,3}, {0,2,3,7}, {0,3,6,7},
//         {3,6,7,10}, {6,7,10,11}, {7,9,10,11},
//         {8,9,10,11}, {4,8,9,11}, {4,5,8,11},
//         {1,4,5,8}, {0,1,4,5}, {0,1,2,4}
//     });

//     for (const auto &cl : checkLists) {
//         bool ok = true;
//         for (int k=0; k<12; k++) {
//             if (std::find(cl.begin(), cl.end(), k) != cl.end()) {
//                 ok &= flag[k];
//             }
//             else {
//                 ok &= !flag[k];
//             }
//         }
//         if (ok) return true;
//     }

//     return false;
// }

int main(int argc, char** argv) {

    ros::init(argc, argv, "extraction_node");
    ros::NodeHandle n("~");

    std::string rgb_left_topic, rgb_right_topic, thermal_topic, laser_topic;
    n.param("rgb_dir",                  rgbDir,                     std::string(""));
    n.param("thermal_dir",              thermalDir,                 std::string(""));
    n.param("laser_dir",                laserDir,                   std::string(""));
    n.param("rgb_left_topic",           rgb_left_topic,             std::string(""));
    n.param("rgb_right_topic",          rgb_right_topic,            std::string(""));
    n.param("thermal_topic",            thermal_topic,              std::string(""));
    n.param("laser_topic",              laser_topic,                std::string(""));
    n.param("IR_min_intensity",         IR_min_intensity,           0.0);
    n.param("IR_max_intensity",         IR_max_intensity,           0.0);
    n.param("laser_min_range",          laser_min_range,            0.1);
    n.param("laser_max_range",          laser_max_range,            100.0);
    n.param("laser_edge_th",            laser_edge_th,              0.5);
    n.param("laser_edge_sample_radius", laser_edge_sample_radius,   3);
    n.param("debug_output",             debug_output,               false);
    n.param("sample_step",              sample_step,                1);

    createCleanFolder(rgbDir);
    createCleanFolder(rgbDir + "/left");
    createCleanFolder(rgbDir + "/right");
    createCleanFolder(thermalDir);
    createCleanFolder(laserDir);
    createCleanFolder(laserDir + "/edges");
    createCleanFolder(laserDir + "/debug");
    
    ros::Subscriber sub_thermal_image = n.subscribe(thermal_topic, 1000, thermal_callback);
    ros::Subscriber sub_rgb_left_image = n.subscribe(rgb_left_topic, 1000, rgb_left_callback);
    ros::Subscriber sub_rgb_right_image = n.subscribe(rgb_right_topic, 1000, rgb_right_callback);
    ros::Subscriber sub_laser_image = n.subscribe<sensor_msgs::PointCloud2>(laser_topic, 1000, laser_callback);
    
    std::thread joint_process;
    joint_process = std::thread(process);
    // joint_process = std::thread(process_RGB_IR);
    ros::spin();
    
    return 0;
}
