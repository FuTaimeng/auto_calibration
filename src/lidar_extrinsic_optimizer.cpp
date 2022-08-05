#include <ros/ros.h>

#include "Eigen/Core"
#include "Eigen/Eigenvalues"

#include "opencv2/core.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/core/eigen.hpp"

#include "openMVG/system/timer.hpp"

// #include <pcl/registration/gicp.h>
#include <pcl/features/normal_3d.h>
#include <pcl/kdtree/kdtree_flann.h>

#include "ceres_ICP.hpp"
#include "model_io.hpp"
#include "file_io.hpp"
#include "pose3d.hpp"


typedef pcl::PointCloud<pcl::PointXYZ> PointCloudT;
typedef pcl::PointCloud<pcl::Normal> NormalCloudT;

void transformPointCloud( const PointCloudT &src,
                          PointCloudT &res,
                          const Mat3 &R, const Vec3 &C );

void sphereClip(const PointCloudT &src,
                PointCloudT &res,
                const double radius);

void getOptimizeTargets(std::string str, std::vector<std::string> &tars);

void getDistanceLimits(std::string str, std::vector<double> &lims);


int main(int argc, char** argv) {
    ros::init(argc, argv, "lidar_extrinsic_optimizer_node");
    ros::NodeHandle n("~");

    std::string imageDir, lidarCloudDir, stereoCloudDir, resultDir, initRDir, initCDir;
    int frameSampleStep, cloudSampleStep, interationTimes, cloudMaxDist;
    std::string optimizeTargetStr, distLimStr;
    n.param("image_dir",            imageDir,               std::string(""));
    n.param("lidar_cloud_dir",      lidarCloudDir,          std::string(""));
    n.param("stereo_cloud_dir",     stereoCloudDir,         std::string(""));
    n.param("result_dir",           resultDir,              std::string(""));
    n.param("init_R_dir",           initRDir,               std::string(""));
    n.param("init_t_dir",           initCDir,               std::string(""));
    n.param("frame_sample_step",    frameSampleStep,        1);
    n.param("cloud_sample_step",    cloudSampleStep,        100);
    n.param("inter_times",          interationTimes,        10);
    n.param("dist_lim",             distLimStr,             std::string("0.5"));
    n.param("optimize_target",      optimizeTargetStr,      std::string("rt"));
    n.param("cloud_max_dist",       cloudMaxDist,           100);

    createCleanFolder(resultDir);

    omp_lock_t writelock;
    omp_init_lock(&writelock);
    openMVG::system::Timer timer;

    std::vector<std::string> leftImageFileNames;
    std::vector<std::string> rightImageFileNames;
    readData(imageDir+"/left_filenames.csv", leftImageFileNames);
    readData(imageDir+"/right_filenames.csv", rightImageFileNames);
    if(leftImageFileNames.size() != rightImageFileNames.size()) {
        std::cerr << "Left and right images are not the same. Check folder: " 
            << imageDir << std::endl;
    } else {
        std::cout << "Found (left-right) " << leftImageFileNames.size() 
            << " image pairs. " << std::endl;
    }

    const int nImagePair = leftImageFileNames.size();
    const int nImage = nImagePair*2;

    Mat P, K, R, C;
    readData(imageDir + "/P1.dat", 3, 3, ' ', P);
    readData(initRDir, 3, 3, ' ', R);
    readData(initCDir, 3, 1, ' ', C);
    K = P.block<3,3>(0,0);
    std::cout << "Load lidar extrinsic done." << std::endl;

    std::vector<std::string> lidarCloudIndexs(nImagePair);
    std::vector<PointCloudT> stereoClouds(nImagePair);
    std::vector<PointCloudT> lidarClouds(nImagePair);
    std::vector<PointCloudT> lidarEdges(nImagePair);
    // std::vector<std::vector<Vec3>> lidarNormals(nImagePair);
    // std::vector<std::vector<bool>> lidarNormalFlags(nImagePair);

    #pragma omp parallel for
    for(int i=0; i<nImagePair; i+=frameSampleStep) {
        int p1 = leftImageFileNames[i].find_last_of('/');
        int p2 = leftImageFileNames[i].find_last_of('.');
        std::string idxStr = leftImageFileNames[i].substr(p1+1, p2-p1-1);
        lidarCloudIndexs[i] = idxStr;

        PointCloudT lidarCloud;
        std::string fname = lidarCloudDir + "/" + idxStr + ".ply";
        if (pcl::io::loadPLYFile(fname, lidarCloud) == -1) {
            std::cerr << "Failed to read: " << fname << std::endl;
            continue;
        }
        sphereClip(lidarCloud, lidarCloud, cloudMaxDist);

        PointCloudT lidarEdge;
        fname = lidarCloudDir + "/edges/" + idxStr + ".ply";
        if (pcl::io::loadPLYFile(fname, lidarEdge) == -1) {
            std::cerr << "Failed to read: " << fname << std::endl;
            continue;
        }
        sphereClip(lidarEdge, lidarEdge, cloudMaxDist);

        // NormalCloudT lidarNormal;
        // pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normalEstimation;
        // normalEstimation.setInputCloud(lidarCloud.makeShared());
        // normalEstimation.setKSearch(20);
        // pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree(new pcl::search::KdTree<pcl::PointXYZ>);
        // normalEstimation.setSearchMethod(kdtree);
        // normalEstimation.compute(lidarNormal);

        // std::vector<Vec3> lidarNormal;
        // std::vector<bool> isValidNormal;
        // computeNormals(lidarCloud, lidarNormal, isValidNormal);

        PointCloudT stereoCloud;
        fname = stereoCloudDir + "/" + std::to_string(i) + ".ply";
        if (pcl::io::loadPLYFile(fname, stereoCloud) == -1) {
            std::cerr << "Failed to read: " << fname << std::endl;
            continue;
        }
        sphereClip(stereoCloud, stereoCloud, cloudMaxDist);

        stereoClouds[i] = std::move(stereoCloud);
        lidarClouds[i] = std::move(lidarCloud);
        lidarEdges[i] = std::move(lidarEdge);
        // lidarNormals[i] = std::move(lidarNormal);
        // lidarNormalFlags[i] = std::move(isValidNormal);
    }
    std::cout << "Load stereo and lidar clouds done." << std::endl;

    writeData(resultDir+"/init_R.dat", R);
    writeData(resultDir+"/init_T.dat", C);

    std::vector<std::string> optimizeTargets;
    getOptimizeTargets(optimizeTargetStr, optimizeTargets);
    while (optimizeTargets.size() < interationTimes) {
        optimizeTargets.push_back(*optimizeTargets.rbegin());
    }

    std::vector<double> distanceLimits;
    getDistanceLimits(distLimStr, distanceLimits);
    while (distanceLimits.size() < interationTimes) {
        distanceLimits.push_back(*distanceLimits.rbegin());
    }

    Mat3 uniformR(R.transpose());
    Vec3 uniformC(-R.transpose()*C);

    timer.reset();
    for (int l=0; l<interationTimes; l++) {
        std::cout << "Processing inter " << l << " ... " << std::endl;
        std::cout << "Optimize target: " << optimizeTargets[l] << std::endl;
        
        const double squDistLim = distanceLimits[l] * distanceLimits[l];
        std::cout << "  Dist lim: " << distanceLimits[l] << std::endl;

        std::vector<Vec3> src_points, dst_points;
        #pragma omp parallel for
        for (int i=0; i<nImagePair; i+=frameSampleStep) {
            const int nPoint = stereoClouds[i].points.size();

            pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
            kdtree.setInputCloud(lidarClouds[i].makeShared());

            for (int j=0; j<nPoint; j+=cloudSampleStep) {
                Vec3 src_pt(stereoClouds[i].points[j].x, 
                            stereoClouds[i].points[j].y, 
                            stereoClouds[i].points[j].z);
                Vec3 trans_pt = uniformR * src_pt + uniformC;
                pcl::PointXYZ trans_pt_xyz(trans_pt(0), trans_pt(1), trans_pt(2));

                const int K = 1;
                std::vector<int> KNNIdx(K);
                std::vector<float> KNNSquDist(K);
                kdtree.nearestKSearch(trans_pt_xyz, K, KNNIdx, KNNSquDist);
                for (int k=0; k<K; k++) {
                    if (KNNSquDist[k] < squDistLim) {
                        Vec3 dst_pt(lidarClouds[i].points[KNNIdx[k]].x, 
                                    lidarClouds[i].points[KNNIdx[k]].y, 
                                    lidarClouds[i].points[KNNIdx[k]].z);
                        
                        omp_set_lock(&writelock);
                        src_points.push_back(src_pt);
                        dst_points.push_back(dst_pt);
                        omp_unset_lock(&writelock);

                        // if (lidarNormalFlags[i][KNNIdx[k]]) {
                        //     Vec3 dst_nor = lidarNormals[i][KNNIdx[k]];
                        //     omp_set_lock(&writelock);
                        //     src_points.push_back(src_pt);
                        //     dst_points.push_back(dst_pt);
                        //     dst_normals.push_back(dst_nor);
                        //     omp_unset_lock(&writelock);
                        // }

                        break;
                    }
                }
            }
        }

        // optimization
        Mat3 newR;
        Vec3 newC;
        ceresICPOptimizer(uniformR, uniformC, src_points, dst_points, newR, newC, optimizeTargets[l]);
        uniformR = newR;
        uniformC = newC;

        Mat3 calibR(uniformR.transpose());
        Vec3 calibC(-uniformR.transpose()*uniformC);
        writeData(resultDir+"/R_"+std::to_string(l)+".dat", calibR);
        writeData(resultDir+"/T_"+std::to_string(l)+".dat", calibC);
    }
    std::cout << "Total optimization cost: " << timer.elapsed() << " seconds." << std::endl;

    Mat3 calibR(uniformR.transpose());
    Vec3 calibC(-uniformR.transpose()*uniformC);

    std::cout << "calibR:\n" << calibR << std::endl;
    std::cout << "calibC:\n" << calibC << std::endl;

    writeData(resultDir+"/R.dat", calibR);
    writeData(resultDir+"/T.dat", calibC);
    writeData(resultDir+"/cloud_indexs.csv", lidarCloudIndexs);
    
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

void sphereClip(const PointCloudT &src,
                PointCloudT &res,
                const double radius) {
    
    PointCloudT filtered;
    for (const auto &pt : src.points) {
        if (pt.x*pt.x+pt.y*pt.y+pt.z*pt.z < radius*radius) {
            filtered.push_back(pt);
        }
    }
    res = std::move(filtered);
}

void getOptimizeTargets(std::string str, std::vector<std::string> &tars) {

    while (str.size() > 0) {
        std::size_t p = str.find_first_of('_');
        std::string part;
        if (p == std::string::npos) {
            part = str; str = "";
        }
        else {
            part = str.substr(0, p);
            str = str.substr(p+1);
        }

        std::size_t q = part.find_first_of('*');
        std::string op; int num;
        if (q == std::string::npos) {
            op = part; num = 1;
        }
        else {
            op = part.substr(0, q);
            num = std::stoi(part.substr(q+1));
        }

        while (num--) {
            tars.push_back(op);
        }
    }
}

void getDistanceLimits(std::string str, std::vector<double> &lims) {
    std::vector<std::string> strNums;
    getOptimizeTargets(str, strNums);
    for (std::string s : strNums) {
        lims.push_back(std::stod(s));
    }
}