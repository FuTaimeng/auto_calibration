#include <ros/ros.h>

#include "Eigen/Core"

#include "opencv2/core.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/core/eigen.hpp"

#include "openMVG/system/timer.hpp"
#include "openMVG/cameras/cameras.hpp"
#include "openMVG/matching/indMatch_utils.hpp"
#include "openMVG/features/regions_factory_io.hpp"

#include "model_io.hpp"
#include "file_io.hpp"
#include "pose3d.hpp"

using namespace openMVG;


int main(int argc, char** argv) {
    ros::init(argc, argv, "stereo_triangulation_node");
    ros::NodeHandle n("~");

    std::string imageDir, matchDir, cloudDir;
    double inlier_lower_th, inlier_upper_th;
    n.param("image_dir",        imageDir,           std::string(""));
    n.param("match_dir",        matchDir,           std::string(""));
    n.param("cloud_dir",        cloudDir,           std::string(""));
    n.param("inlier_lower_th",  inlier_lower_th,    1.0);
    n.param("inlier_upper_th",  inlier_upper_th,    10.0);

    createCleanFolder(cloudDir);

    openMVG::system::Timer timer;
    omp_lock_t writelock;
    omp_init_lock(&writelock);

    // ------------------------------------
    // 1. load stereo image list
    // ------------------------------------
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

    cv::Mat tmp_img = cv::imread(leftImageFileNames[0]);
    const int imgWidth = tmp_img.cols;
    const int imgHeight = tmp_img.rows;

    int nImagePair = leftImageFileNames.size();
    int nImage = nImagePair*2;

    std::cout << "Load image done in " << timer.elapsed() << " seconds. " << std::endl;

    // ------------------------------------
    // 2. load intrinsic and extrinsic
    // ------------------------------------
    timer.reset();
    Mat P1, P2;
    readData(imageDir + "/P1.dat", 3, 4, ' ', P1);
    readData(imageDir + "/P2.dat", 3, 4, ' ', P2);
    Mat K1 = P1.block<3,3>(0,0);
    Mat K2 = P2.block<3,3>(0,0);
    Mat R  = Eigen::Matrix3d::Identity();
    Mat C  = K2.inverse() * P2.block<3,1>(0,3);
    std::cout << "Load intrinsic and extrinsic done in " << timer.elapsed() 
        << " seconds. " << std::endl;

    // ------------------------------------
    // 3. load matches and features
    // ------------------------------------
    timer.reset();
    matching::PairWiseMatches matches;
    if(!Load(matches, matchDir + "/matches.bin")) {
        std::cerr << std::endl << "Invalid matches file." << std::endl;
        return EXIT_FAILURE;
    }
    // std::cout << matches.size() << std::endl;
    // for (const auto &match : matches) {
    //     std::cout << match.first.first << "-" << match.first.second << " size " 
    //         << match.second.size() << std::endl;
    // }
    std::cout << "Load matches done in " << timer.elapsed() << " seconds. " << std::endl;

    timer.reset();
    std::map<int, features::PointFeatures> regions_perImage;
    #pragma omp parallel for
    for(int i=0; i<nImage; i++) {
        std::string featFile = matchDir+"/image_"+std::to_string(i)+".feat";
        std::string descFile = matchDir+"/image_"+std::to_string(i)+".desc";
        std::unique_ptr<features::Regions> regions_ptr(new features::SIFT_Regions());

        if (!regions_ptr->Load(featFile, descFile)) {
            std::cerr << "Invalid regions files for the view: " << i << std::endl;
            EXIT_FAILURE;
        }

        const features::PointFeatures &feats = regions_ptr->GetRegionsPositions();
        omp_set_lock(&writelock);
        regions_perImage.insert(std::make_pair(i, feats));
        std::cout << "    loading features: " << regions_perImage.size() << "/" 
            << nImage << "\r" << std::flush;
        omp_unset_lock(&writelock);
    }
    std::cout << std::endl;
    std::cout << "Load features done in " << timer.elapsed() << " seconds. " << std::endl;

    // ------------------------------------
    // 4. triangulation and save cloud
    // ------------------------------------
    std::cout << "Inlier range: " << inlier_lower_th << " - " << inlier_upper_th << std::endl;

    double total_reproj_err = 0;
    int total_reproj_cnt = 0;

    int count = 0;
    #pragma omp parallel for
    for (int n=0; n<nImagePair; n++) {
        int I = n*2, J = n*2+1;
        Pair stereo_pair(I,J);
        
        if (matches.count(stereo_pair) == 0) {
            omp_set_lock(&writelock);
            std::cerr << "No matches in stereo pair " << n << std::endl;
            omp_unset_lock(&writelock);
            continue;
        }
        
        const matching::IndMatches &match_pairs = matches.at(stereo_pair);
        size_t number_matches = match_pairs.size();

        std::shared_ptr<cameras::IntrinsicBase> cam_I, cam_J;
        cam_I = std::make_shared<cameras::Pinhole_Intrinsic>
                (imgWidth, imgHeight, K1(0,0), K1(0,2), K1(1,2));
        cam_J = std::make_shared<cameras::Pinhole_Intrinsic>
                (imgWidth, imgHeight, K2(0,0), K2(0,2), K2(1,2));

        std::vector<cv::Point2d> cvX1, cvX2;
        for (int i=0; i<number_matches; i++) {
            const auto &match = match_pairs[i];

            Vec2 x1, x2;
            x1 = cam_I->get_ud_pixel(regions_perImage[I][match.i_].coords().cast<double>());
            x2 = cam_J->get_ud_pixel(regions_perImage[J][match.j_].coords().cast<double>());

            cvX1.push_back(cv::Point2d(x1(0), x1(1)));
            cvX2.push_back(cv::Point2d(x2(0), x2(1)));
        }

        cv::Mat cvP1, cvP2, cvX_h;
        cv::eigen2cv(P1, cvP1);
        cv::eigen2cv(P2, cvP2);
        cv::triangulatePoints(cvP1, cvP2, cv::Mat(cvX1), cv::Mat(cvX2), cvX_h);
        
        std::vector<cv::Point3d> cvX;
        std::vector<cv::Scalar> colors;
            
        for(size_t i=0; i<number_matches; i++) {
            if(std::fabs(cvX_h.at<double>(3,i)) < 1e-6) continue;

            // Cheirality test & depth cut
            double depthI = cvX_h.at<double>(2,i) / cvX_h.at<double>(3, i);
            if (depthI < 0) continue;

            cv::Point3d pt( cvX_h.at<double>(0,i) / cvX_h.at<double>(3,i),
                            cvX_h.at<double>(1,i) / cvX_h.at<double>(3,i),
                            cvX_h.at<double>(2,i) / cvX_h.at<double>(3,i) );
            double d = std::sqrt(pt.x*pt.x + pt.y*pt.y + pt.z*pt.z);

            if (d > inlier_lower_th && d < inlier_upper_th) {
                cvX.push_back(pt);

                Vec3 p3d(pt.x, pt.y, pt.z);
                Vec3 lp = K1 * p3d;
                double u = lp(0)/lp(2), v = lp(1)/lp(2);
                double reproj_err = sqrt(pow(cvX1[i].x-u, 2) + pow(cvX1[i].y-v, 2));
                // std::cout << reproj_err << std::endl;
                total_reproj_err += reproj_err;
                total_reproj_cnt ++;

                colors.push_back(cv::Scalar(255, 255, 255));
            }
        }

        outputPointsToPLY(cvX, colors, cloudDir+"/"+std::to_string(n)+".ply");

        omp_set_lock(&writelock);
        std::cout << "    processed " << ++count << "/" << nImagePair << "\r" << std::flush;
        omp_unset_lock(&writelock);
    }
    std::cout << std::endl;
    std::cout << "Triangulation and save cloud done in " << timer.elapsed() 
        << " seconds. " << std::endl;

    
    std::cout << "reproj err: " << total_reproj_err / total_reproj_cnt << std::endl;

    return 0;
}