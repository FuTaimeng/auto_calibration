#include <ros/ros.h>

#include "Eigen/Core"

#include "opencv2/core.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/core/eigen.hpp"

#include "openMVG/system/timer.hpp"

#include "model_io.hpp"
#include "file_io.hpp"
#include "pose3d.hpp"
#include "edge_filter.hpp"

#include "direct_BA.hpp"

typedef pcl::PointXYZ PointT;
typedef pcl::PointCloud<PointT> PointCloudT;


void cloudFilter(const PointCloudT &cloud, 
                 const Mat &K, const Mat &R, const Mat &C, 
                 const cv::Mat &tractionMap,
                 const int threshold,
                 PointCloudT &inlier,
                 std::vector<bool> &flags);

void getSelectedData( const std::vector<int> &selIndex, 
                      const int cloudSampleStep,
                      const std::vector<PointCloudT> &clouds, 
                      const std::vector<cv::Mat> &tractionMaps, 
                      std::vector<std::vector<Vec3>> &selPoints, 
                      std::vector<std::vector<double>> &selTargets, 
                      std::vector<cv::Mat> &selTractions );

void evalCostOverAll( const Mat3 &K, const Mat3 &R, const Vec3 &C,
                      const std::vector<PointCloudT> &clouds,
                      const std::vector<cv::Mat> &tractionMaps,
                      std::vector<double> &costs,
                      double &avgCost );

double evalCostOverSel( const std::vector<int> &selIndex,
                        const int cloudSampleStep,
                        const Mat3 &K, const Mat3 &R, const Vec3 &C,
                        const std::vector<PointCloudT> &clouds,
                        const std::vector<cv::Mat> &tractionMaps );

void getOptimizeTargets(std::string str, std::vector<std::string> &tars);

void roughOptimizeTrans(const std::vector<int> &selIndex, 
                        const int cloudSampleStep,
                        const int radius, const double bias, const double biasLim,
                        const Mat3 &K, const Mat3 &R, const Vec3 &C, 
                        const std::vector<PointCloudT> &clouds,
                        const std::vector<cv::Mat> &tractionMaps,
                        const int outlierDistLim, 
                        Vec3 &newC);

void roughOptimizeRot(const std::vector<int> &selIndex, 
                      const int radius, const double bias, const double biasLim,
                      const Mat3 &K, const Mat3 &R, const Vec3 &C, 
                      const std::vector<PointCloudT> &clouds,
                      const std::vector<cv::Mat> &tractionMaps,
                      const int outlierDistLim, 
                      Mat3 &newR);

void edgeAlignmentOptimize( const std::vector<int> &selIndex,
                            const int cloudSampleStep,
                            const Mat3 &K, const Mat3 &R, const Vec3 &C, 
                            const std::vector<PointCloudT> &clouds,
                            const std::vector<cv::Mat> &tractionMaps,
                            const int outlierDistLim,
                            const int ceresInterations,
                            const std::string optimizeTarget,
                            Mat3 &newR, Vec3 &newC );

void reprojectCloudFilter(const int gridSize, 
                          const int img_w, const int img_h,
                          const Mat3 &K, const Mat3 &R, const Vec3 &C, 
                          const PointCloudT &inCloud, PointCloudT &outCloud);


int main(int argc, char** argv) {
    ros::init(argc, argv, "thermal_extrinsic_optimizer_node");
    ros::NodeHandle n("~");

    std::string imageDir, cloudDir, resultDir;
    std::string initRDir, initCDir;
    std::string optimizeTargetStr;
    bool useCachedTractionMap;  // should find the cache in */image/traction
    bool useCVDist;         // traction map method choice
    int outlierDistLim;     // dist threshold to filter
    int ceresInterations;   // max # of optimize interation
    int frameSampleStep;    // sample step of the optimizer
    int cloudSampleStep;    // sample step of the optimizer
    int interationTimes;    // # of interations

    n.param("image_dir",        imageDir,               std::string(""));
    n.param("cloud_dir",        cloudDir,               std::string(""));
    n.param("result_dir",       resultDir,              std::string(""));
    n.param("init_R_dir",       initRDir,               std::string(""));
    n.param("init_t_dir",       initCDir,               std::string(""));
    n.param("optimize_target",  optimizeTargetStr,      std::string("rt"));
    n.param("cached_traction",  useCachedTractionMap,   false);
    n.param("use_cvdist",       useCVDist,              true);
    n.param("outlier_dist_lim", outlierDistLim,         150);
    n.param("ceres_inter",      ceresInterations,       30);
    n.param("frame_sample_step",frameSampleStep,        2);
    n.param("cloud_sample_step",cloudSampleStep,        10);
    n.param("inter_times",      interationTimes,        1);

    createCleanFolder(resultDir);

    openMVG::system::Timer timer;
    omp_lock_t writelock;
    omp_init_lock(&writelock);

    // ------------------------------------
    // 1. load thermal images
    // ------------------------------------
    std::vector<std::string> thermalImageFileNames;
    readData(imageDir+"/thermal_filenames.csv", thermalImageFileNames);
    int nImage = thermalImageFileNames.size();
    std::cout << "Found " << nImage << " thermal images. " << std::endl;

    std::vector<cv::Mat> images(nImage);
    int count = 0;
    #pragma omp parallel for
    for(size_t i=0; i<nImage; i++) {
        cv::Mat image = cv::imread(thermalImageFileNames[i], -1);
        omp_set_lock(&writelock);
        std::cout << "Load image " << ++count << "/" << nImage << "\r" << std::flush;
        images[i] = std::move(image);
        omp_unset_lock(&writelock);
    }
    std::cout << std::endl;
    std::cout << "Load images done in " << timer.elapsed() << " seconds. "  << std::endl;
    
    // ------------------------------------
    // 2. load intrinsic and extrinsic
    // ------------------------------------
    timer.reset();
    Mat K, R, C;
    readData(imageDir + "/Kt.dat", 3, 3, ' ', K);
    readData(initRDir, 3, 3, ' ', R);
    readData(initCDir, 3, 1, ' ', C);
    std::cout << "Load thermal intrinsic and extrinsic done in " << timer.elapsed() 
        << " seconds. " << std::endl;

    // ------------------------------------
    // 3. load point clouds
    // ------------------------------------
    timer.reset();
    std::vector<PointCloudT> clouds(nImage);
    std::vector<PointCloudT> dsClouds(nImage);
    count = 0;
    #pragma omp parallel for
    for (int i=0; i<nImage; i++) {
        int p1 = thermalImageFileNames[i].find_last_of('/');
        int p2 = thermalImageFileNames[i].find_last_of('.');
        std::string idxStr = thermalImageFileNames[i].substr(p1+1, p2-p1-1);
        std::string fname(cloudDir+"/"+idxStr+".ply");
        PointCloudT cloud;
        if (!boost::filesystem::exists(fname) || 
            pcl::io::loadPLYFile<PointT>(fname, cloud) == -1) {

            std::cerr << "Failed to read: " << fname << std::endl;
            continue;
        }

        PointCloudT dsCloud;
        reprojectCloudFilter(5, images[0].cols, images[0].rows, K, R, C, cloud, dsCloud);
        // std::cout << cloud.points.size() << "  " << dsCloud.points.size() << std::endl; 

        omp_set_lock(&writelock);
        std::cout << "Load cloud " << ++count << "/" << nImage << "\r" << std::flush;
        clouds[i] = std::move(cloud);
        dsClouds[i] = std::move(dsCloud);
        omp_unset_lock(&writelock);
    }
    std::cout << std::endl;
    std::cout << "Load point clouds done in " << timer.elapsed() << " seconds. " << std::endl;

    // ------------------------------------
    // 4. generate traction maps
    // ------------------------------------
    bool tractionMapsLoadedSuccessfully = false;
    std::vector<cv::Mat> tractionMaps(nImage);

    timer.reset();
    if (useCachedTractionMap && boost::filesystem::exists(imageDir+"/traction")) {
        int count = 0;
        #pragma omp parallel for
        for(size_t i=0; i<nImage; i++) {
            cv::Mat tractionMap = cv::imread(imageDir+"/thermal_traction/"+std::to_string(i)+".png", CV_8UC1);
            omp_set_lock(&writelock);
            std::cout << "Load traction map " << ++count << "/" << nImage << "\r" << std::flush;
            tractionMaps[i] = std::move(tractionMap);
            omp_unset_lock(&writelock);
        }
        std::cout << std::endl;
        std::cout << "Load traction map done in " << timer.elapsed() << " seconds. "  << std::endl;
    } 
    else {
        count = 0;
        #pragma omp parallel for
        for (int i=0; i<nImage; i++) {
            cv::Mat edge, tmp, tractionMap;

            if (useCVDist) {
                cv::Canny(images[i], tmp, 10, 30);
                contourEdgeFilter(tmp, edge, 50, 100);
                edge = 255 - edge;
            }
            else {
                cv::Mat xgrad, ygrad;
                cv::Sobel(images[i], xgrad, CV_16S, 1, 0, 3);
                cv::Sobel(images[i], ygrad, CV_16S, 0, 1, 3);
                cv::convertScaleAbs(xgrad, xgrad);
                cv::convertScaleAbs(ygrad, ygrad);
                cv::addWeighted(xgrad, 0.5, ygrad, 0.5, 0, edge);
                // cv::threshold(edge, edge, 30, 255, CV_THRESH_BINARY_INV);
            }

            if (useCVDist) {
                cv::distanceTransform(edge, tmp, CV_DIST_L2, 5);
                tmp = tmp * 10;
                tmp.convertTo(tmp, CV_8UC1);
                tractionMap = 255 - tmp;
            }
            else {
                generateTractionMap(edge, tractionMap);
            }

            omp_set_lock(&writelock);
            std::cout << "Generate traction map " << ++count << "/" << nImage << "\r" << std::flush;
            tractionMaps[i] = std::move(tractionMap);
            omp_unset_lock(&writelock);
        }
        std::cout << std::endl;
        std::cout << "Generate traction map done in " << timer.elapsed() << " seconds. "  << std::endl;
    }

    // ------------------------------------
    // 6. optimize extrinsic
    // ------------------------------------
    std::cout << "Interation : " << interationTimes << std::endl;
    std::cout << "Optimize target : " << optimizeTargetStr << std::endl;
    std::cout << "Cloud filter threshold : " << outlierDistLim << std::endl;

    // initialize selected index
    std::vector<int> selIndex;
    if (boost::filesystem::exists(imageDir+"/frame_sel.txt")) {
        std::vector<string> selIndexStr;
        readData(imageDir+"/frame_sel.txt", selIndexStr);
        for (const string &s : selIndexStr)
            if (s.length() > 0)
                selIndex.push_back(std::stoi(s));
        std::cout << "Sel file length : " << selIndex.size() << std::endl;
    }
    else {
        for (int i=0; i<nImage; i+=frameSampleStep)
            selIndex.push_back(i);
        std::cout << "Frame sample step : " << frameSampleStep << std::endl;
    }

    // initialize optimize targets
    std::vector<std::string> optimizeTargets;
    getOptimizeTargets(optimizeTargetStr, optimizeTargets);
    while (optimizeTargets.size() < interationTimes) {
        optimizeTargets.push_back(*optimizeTargets.rbegin());
    }

    Mat3 uniformR, roughR, newR;
    Vec3 uniformC, roughC, newC;

    writeData(resultDir+"/init_R.dat", R);
    writeData(resultDir+"/init_T.dat", C);

    // rough optimize
    timer.reset();
    roughR = R;
    roughC = C;
    roughOptimizeRot(selIndex, 6, 1, 6, K, roughR, roughC, dsClouds, tractionMaps, outlierDistLim, newR);
    roughR = newR;

    std::cout << "Rough rot calib in " << timer.elapsed() << " seconds. "  << std::endl;
    writeData(resultDir+"/rough_R.dat", roughR);

    timer.reset();
    roughOptimizeTrans(selIndex, cloudSampleStep, 2, 0.04, 0.12, K, roughR, roughC, clouds, tractionMaps, outlierDistLim, newC);
    roughC = newC;
    roughOptimizeTrans(selIndex, cloudSampleStep, 2, 0.02, 0.06, K, roughR, roughC, clouds, tractionMaps, outlierDistLim, newC);
    roughC = newC;

    std::cout << "Rough trans calib done in " << timer.elapsed() << " seconds. "  << std::endl;
    writeData(resultDir+"/rough_T.dat", roughC);

    // fine optimize
    uniformR = roughR;
    uniformC = roughC;

    for (int l=0; l<interationTimes; l++) {
        std::cout << "Processing inter " << l << " ... " << std::endl;

        timer.reset();
        edgeAlignmentOptimize(selIndex, cloudSampleStep, K, uniformR, uniformC, clouds, 
            tractionMaps, outlierDistLim, ceresInterations, optimizeTargets[l], newR, newC);
        uniformR = newR;
        uniformC = newC;

        std::cout << "Optimizer done in " << timer.elapsed() << " seconds. "  << std::endl;
        writeData(resultDir+"/R_"+std::to_string(l)+".dat", uniformR);
        writeData(resultDir+"/T_"+std::to_string(l)+".dat", uniformC);
    }

    std::cout << "Before:" << std::endl;
    std::cout << R << std::endl << C << std::endl;
    std::cout << "After:" << std::endl;
    std::cout << uniformR << std::endl << uniformC << std::endl;

    writeData(resultDir+"/R.dat", uniformR);
    writeData(resultDir+"/T.dat", uniformC);

    return 0;
}

void reprojectOntoImage(const PointCloudT &cloud, 
                        const Mat &K, const Mat &R, const Mat &C, 
                        cv::Mat &image, 
                        double radius, cv::Scalar color) {
    for (const auto &pos : cloud.points) {
        Vec3 X, x;
        X << pos.x, pos.y, pos.z;
        x = K * (R * X + C);

        if (x(2) <= 0) continue;

        cv::Point uv(x(0)/x(2), x(1)/x(2));
        if (uv.x>=0 && uv.x<image.cols && uv.y>=0 && uv.y<image.rows) {
            cv::circle(image, uv, radius, color, radius);
            // std::cout<<uv<<std::endl;
        }
    }
}

void reprojectOntoImage(const PointCloudT &cloud, 
                        const Mat &K, const Mat &R, const Mat &C, 
                        const std::vector<bool> &flags,
                        cv::Mat &image, 
                        double radiusT, cv::Scalar colorT,
                        double radiusF, cv::Scalar colorF) {

    const auto &points = cloud.points;
    for (int i=0; i<points.size(); i++) {
        const auto &pt = points[i];

        Vec3 X, x;
        X << pt.x, pt.y, pt.z;
        x = K * (R * X + C);

        if (x(2) <= 0) continue;

        cv::Point uv(x(0)/x(2), x(1)/x(2));
        if (uv.x>=0 && uv.x<image.cols && uv.y>=0 && uv.y<image.rows) {
            if (flags[i]) {
                if (radiusT > 0)
                    cv::circle(image, uv, 0.5, colorT, 2);
            } else {
                if (radiusF > 0)
                    cv::circle(image, uv, 0.5, colorF, 2);
            }
            // cv::circle(image, uv, 0.5, colorT, 2);
        }
    }
}

void cloudFilter(const PointCloudT &cloud, 
                 const Mat &K, const Mat &R, const Mat &C, 
                 const cv::Mat &tractionMap,
                 const int threshold,
                 PointCloudT &inliers,
                 std::vector<bool> &flags) {

    inliers.clear();
    flags.clear();
    const auto &points = cloud.points;
    flags.reserve(points.size());

    for (int i=0; i<points.size(); i++) {
        const auto &pt = points[i];

        Vec3 X, x;
        X << pt.x, pt.y, pt.z;
        x = K * (R * X + C);
        int u = x(0)/x(2);
        int v = x(1)/x(2);

        flags[i] = false;
        if (x(2)>0 && u>=0 && u<tractionMap.cols && v>=0 && v<tractionMap.rows) {
            if (255 - int(tractionMap.at<uchar>(v, u)) <= threshold) {
                flags[i] = true;
                inliers.push_back(pt);
            }
        }
    }
}

void getSelectedData( const std::vector<int> &selIndex, 
                      const int cloudSampleStep,
                      const std::vector<PointCloudT> &clouds, 
                      const std::vector<cv::Mat> &tractionMaps, 
                      std::vector<std::vector<Vec3>> &selPoints, 
                      std::vector<std::vector<double>> &selTargets, 
                      std::vector<cv::Mat> &selTractions ) {

    int N = selIndex.size();
    selPoints.resize(N);
    selTargets.resize(N);
    selTractions.resize(N);

    for (int i=0; i<N; i++) {
        int idx = selIndex[i];
        for (int j=0; j<clouds[idx].points.size(); j+=cloudSampleStep) {
            const auto &pt = clouds[idx].points[j];
            selPoints[i].push_back(Vec3(pt.x, pt.y, pt.z));
            selTargets[i].push_back(255);
        }
        selTractions[i] = tractionMaps[idx];
    }
}

void evalCostOverAll( const Mat3 &K, const Mat3 &R, const Vec3 &C,
                      const std::vector<PointCloudT> &clouds,
                      const std::vector<cv::Mat> &tractionMaps,
                      std::vector<double> &costs,
                      double &avgCost ) {

    int N = clouds.size();

    std::vector<std::vector<Vec3>> points(N);
    std::vector<std::vector<double>> targets(N);
    #pragma omp parallel for
    for (int i=0; i<N; i++) {
        for (const auto &pt : clouds[i].points) {
            points[i].push_back(Vec3(pt.x, pt.y, pt.z));
            targets[i].push_back(255);
        }
    }

    costs = evalCost(K, R, C, points, targets, tractionMaps);

    double totCost = 0, totWeight = 0;
    for (int i=0; i<N; i++) {
        int nPoint = clouds[i].points.size();
        totCost += costs[i] * nPoint;
        totWeight += nPoint;
    }
    avgCost = totCost / totWeight;
}

double evalCostOverSel( const std::vector<int> &selIndex,
                        const int cloudSampleStep,
                        const Mat3 &K, const Mat3 &R, const Vec3 &C,
                        const std::vector<PointCloudT> &clouds,
                        const std::vector<cv::Mat> &tractionMaps ) {

    int N = selIndex.size();

    std::vector<std::vector<Vec3>> points(N);
    std::vector<std::vector<double>> targets(N);
    std::vector<cv::Mat> tractions;
    getSelectedData(selIndex, cloudSampleStep, clouds, tractionMaps, points, targets, tractions);

    auto costs = evalCost(K, R, C, points, targets, tractions);

    double totCost = 0, totWeight = 0;
    for (int k=0; k<N; k++) {
        int i = selIndex[k];
        int nPoint = clouds[i].points.size();
        totCost += costs[k] * nPoint;
        totWeight += nPoint;
    }
    return totCost / totWeight;
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

void roughOptimizeTrans(const std::vector<int> &selIndex, 
                        const int cloudSampleStep,
                        const int radius, const double bias, const double biasLim,
                        const Mat3 &K, const Mat3 &R, const Vec3 &C, 
                        const std::vector<PointCloudT> &clouds,
                        const std::vector<cv::Mat> &tractionMaps,
                        const int outlierDistLim, 
                        Vec3 &newC) {

    const int nImage = tractionMaps.size();
    const int nSelected = selIndex.size();

    std::vector<std::vector<bool>> inlierFlags(nImage);
    std::vector<PointCloudT> inlierClouds(nImage);

    double minCost = 1e5;
    int initInlierCnt = 0;
    int best_ijkn[4];

    #pragma omp parallel for
    for (int l=0; l<nSelected; l++) {
        int idx = selIndex[l];

        std::vector<bool> flags;
        PointCloudT inliers;
        cloudFilter(clouds[idx], K, R, C, tractionMaps[idx], outlierDistLim, inliers, flags);
        
        initInlierCnt += inliers.size();
        // inlierClouds[idx] = std::move(inliers);
        // inlierFlags[idx] = std::move(flags);
    }

    for (int i=-radius; i<=radius; i++) {
        for (int j=-radius; j<=radius; j++) {
            // for (int k=-radius; k<=radius; k++) {
            for (int k=0; k<=0; k++) {
                Vec3 dC, bC;
                dC << i*bias, j*bias, k*bias;
                if (dC.norm() > biasLim) continue;
                bC = C + dC;

                // cloud filter
                int inlierCnt = 0;
                #pragma omp parallel for
                for (int l=0; l<nSelected; l++) {
                    int idx = selIndex[l];

                    std::vector<bool> flags;
                    PointCloudT inliers;
                    cloudFilter(clouds[idx], K, R, bC, tractionMaps[idx], outlierDistLim, inliers, flags);
                    
                    inlierCnt += inliers.size();
                    inlierClouds[idx] = std::move(inliers);
                    inlierFlags[idx] = std::move(flags);
                }

                if (inlierCnt < initInlierCnt) continue;

                double cost = evalCostOverSel(selIndex, cloudSampleStep, K, R, bC, inlierClouds, tractionMaps);

                if (cost < minCost) {
                    minCost = cost;
                    newC = bC;

                    best_ijkn[0]=i; best_ijkn[1]=j; best_ijkn[2]=k; best_ijkn[3]=inlierCnt;
                }
                
                std::cout << "i=" << i << " j=" << j << " k=" << k << " inl=" << inlierCnt << " cost=" << cost << std::endl;
            }
        }
    }

    std::cout << "best : i=" << best_ijkn[0] << " j=" << best_ijkn[1] << " k=" << best_ijkn[2] 
        << " cnt=" << best_ijkn[3] << " cost=" << minCost << std::endl;
}

void roughOptimizeRot(const std::vector<int> &selIndex, 
                      const int radius, const double bias, const double biasLim,
                      const Mat3 &K, const Mat3 &R, const Vec3 &C, 
                      const std::vector<PointCloudT> &clouds,
                      const std::vector<cv::Mat> &tractionMaps,
                      const int outlierDistLim, 
                      Mat3 &newR) {

    const int nImage = tractionMaps.size();
    const int nSelected = selIndex.size();

    int maxCnt = 0;
    int best_ijkn[4];

    for (int i=-radius; i<=radius; i++) {
        for (int j=-radius; j<=radius; j++) {
            for (int k=-radius; k<=radius; k++) {
                Vec3 dR; Mat3 bR;
                dR << i*bias, j*bias, k*bias;
                if (dR.norm() > biasLim) continue;
                dR *= M_PI / 180.0;
                cv::Mat cvdR, cvbR;
                cv::eigen2cv(dR, cvdR);
                cv::Rodrigues(cvdR, cvbR);
                cv::cv2eigen(cvbR, bR);
                bR = bR * R;

                // cloud filter
                int inlierCnt = 0;
                #pragma omp parallel for
                for (int l=0; l<nSelected; l++) {
                    int idx = selIndex[l];

                    std::vector<bool> flags;
                    PointCloudT inliers;
                    cloudFilter(clouds[idx], K, bR, C, tractionMaps[idx], outlierDistLim, inliers, flags);
                    
                    inlierCnt += inliers.size();
                }

                if (inlierCnt > maxCnt) {
                    maxCnt = inlierCnt;
                    newR = bR;
                    best_ijkn[0]=i; best_ijkn[1]=j; best_ijkn[2]=k; best_ijkn[3]=inlierCnt;
                }

                // std::cout << "i=" << i << " j=" << j << " k=" << k << " cnt=" << inlierCnt << std::endl;
            }
        }
    }

    std::cout << "best : i=" << best_ijkn[0] << " j=" << best_ijkn[1] 
        << " k=" << best_ijkn[2] << " cnt=" << best_ijkn[3] << std::endl;
}

void edgeAlignmentOptimize( const std::vector<int> &selIndex,
                            const int cloudSampleStep,
                            const Mat3 &K, const Mat3 &R, const Vec3 &C, 
                            const std::vector<PointCloudT> &clouds,
                            const std::vector<cv::Mat> &tractionMaps,
                            const int outlierDistLim,
                            const int ceresInterations,
                            const std::string optimizeTarget,
                            Mat3 &newR, Vec3 &newC ) {

    const int nImage = tractionMaps.size();
    const int nSelected = selIndex.size();

    std::vector<std::vector<bool>> inlierFlags(nImage);
    std::vector<PointCloudT> inlierClouds(nImage);

    // cloud filter
    #pragma omp parallel for
    for (int k=0; k<nSelected; k++) {
        int i = selIndex[k];

        std::vector<bool> flags;
        PointCloudT inliers;
        cloudFilter(clouds[i], K, R, C, tractionMaps[i], outlierDistLim, inliers, flags);

        inlierClouds[i] = std::move(inliers);
        inlierFlags[i] = std::move(flags);
    }

    std::vector<std::vector<Vec3>> selPoints;
    std::vector<std::vector<double>> selTargets;
    std::vector<cv::Mat> selTractions;
    getSelectedData(selIndex, cloudSampleStep, inlierClouds, tractionMaps, selPoints, selTargets, selTractions);

    directMethordBA(K, R, C, selPoints, selTargets, selTractions, newR, newC, ceresInterations, optimizeTarget);
}

void reprojectCloudFilter(const int gridSize, 
                          const int img_w, const int img_h,
                          const Mat3 &K, const Mat3 &R, const Vec3 &C, 
                          const PointCloudT &inCloud, PointCloudT &outCloud) {
    
    const auto &points = inCloud.points;
    std::map<std::pair<int, int>, PointT> grids;

    for (int i=0; i<points.size(); i++) {
        const auto &pt = points[i];

        Vec3 X, x;
        X << pt.x, pt.y, pt.z;
        x = K * (R * X + C);
        int u = x(0)/x(2);
        int v = x(1)/x(2);

        if (x(2)>0 && u>=0 && u<img_w && v>=0 && v<img_h) {
            auto gid = std::make_pair(u/gridSize, v/gridSize);
            grids[gid] = pt;
        }
    }

    outCloud.clear();
    for (auto it=grids.begin(); it!=grids.end(); it++) {
        outCloud.push_back(it->second);
    }
}