#include <ros/ros.h>

#include <boost/function.hpp>
#include <boost/filesystem.hpp>

#include "openMVG/robust_estimation/robust_estimator_ACRansacKernelAdaptator.hpp"
#include "openMVG/features/sift/SIFT_Anatomy_Image_Describer_io.hpp"
#include "openMVG/robust_estimation/robust_estimator_ACRansac.hpp"
#include "openMVG/features/akaze/image_describer_akaze_io.hpp"
#include "openMVG/multiview/solver_fundamental_kernel.hpp"
#include "openMVG/numeric/eigen_alias_definition.hpp"
#include "openMVG/features/regions_factory_io.hpp"
#include "openMVG/matching/regions_matcher.hpp"
#include "openMVG/matching/indMatch_utils.hpp"
#include "openMVG/matching/svg_matches.hpp"
#include "openMVG/image/image_concat.hpp"
#include "openMVG/image/image_io.hpp"
#include "openMVG/tracks/tracks.hpp"
#include "openMVG/system/timer.hpp"

#include "opencv2/core/eigen.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/core.hpp"

#include "file_io.hpp"
#include "edge_filter.hpp"

using namespace openMVG;
using namespace openMVG::image;
using namespace openMVG::matching;
using namespace openMVG::robust;
using namespace openMVG::tracks;


int main(int argc, char** argv) {
    ros::init(argc, argv, "stero_pair_feature_matcher_node");
    ros::NodeHandle n("~");

    std::string imageDir, matchDir, featDensity, featMethod;
    int outlierDistLim;
    bool edgeCheck, debugOutput;
    n.param("image_dir",            imageDir,               std::string(""));
    n.param("match_dir",            matchDir,               std::string(""));
    n.param("feat_density",         featDensity,            std::string("normal"));
    n.param("edge_mask",            edgeCheck,              false);
    n.param("outlier_dist_lim",     outlierDistLim,         150);
    n.param("debug_output",         debugOutput,            false);
    n.param("feature_mathod",       featMethod,             std::string("sift"));

    if(!boost::filesystem::exists(imageDir)) {
        std::cerr << "image directory: " << imageDir << " does not exist. " << std::endl;
        return -1;
    }
    createFolder(matchDir);
    if (debugOutput) {
        createCleanFolder(matchDir+"/debug");
    }

    openMVG::system::Timer timer;
    omp_lock_t writelock;
    omp_init_lock(&writelock);

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

    int nImagePair = leftImageFileNames.size();
    int nImage = nImagePair*2;

    PairWiseMatches pairwiseMatches;

    std::cout << "Edge maks: " << edgeCheck << std::endl;
    std::cout << "Outlier dist lim: " << outlierDistLim << std::endl;

    timer.reset();
    int count = 0;
    #pragma omp parallel for num_threads(6) schedule(dynamic)
    for (int i=0; i<nImagePair; i++) {
        omp_set_lock(&writelock);
        std::cout << i << " start" << std::endl;
        omp_unset_lock(&writelock);

        // ------------------------------------
        // 1. load images
        // ------------------------------------
        std::vector<Image<unsigned char>> imagePair(2);
        ReadImage(leftImageFileNames[i].c_str(), &imagePair[0]);
        ReadImage(rightImageFileNames[i].c_str(), &imagePair[1]);
        assert(imagePair[0].data());
        assert(imagePair[1].data());

        const int imgWidth = imagePair[0].Width();
        const int imgHeight = imagePair[0].Height();

        omp_set_lock(&writelock);
        std::cout << i << " load img done" << std::endl;
        omp_unset_lock(&writelock);

        std::vector<std::unique_ptr<features::Regions>> regions(2);
        if (featMethod == "mldb") {
            regions[0].reset(new features::AKAZE_Image_describer_MLDB::Regions_type());
            regions[1].reset(new features::AKAZE_Image_describer_MLDB::Regions_type());
        }
        else if (featMethod == "surf") {
            regions[0].reset(new features::AKAZE_Image_describer_SURF::Regions_type());
            regions[1].reset(new features::AKAZE_Image_describer_SURF::Regions_type());
        }
        else {
            regions[0].reset(new features::SIFT_Regions());
            regions[1].reset(new features::SIFT_Regions());
        }

        std::vector<cv::Mat> edgePair(2), tracPair(2);
        Image<unsigned char> mask[2];

        if (edgeCheck) {
            for (int j=0; j<2; j++) {
                Image<unsigned char> &image = imagePair[j];

                cv::Mat cvImage, cvMask, cvEdge, cvTrac;
                cv::eigen2cv(image.GetMat(), cvImage);
                
                double scaleX = 570.0 / imgWidth;
                double scaleY = 430.0 / imgHeight;
                cv::resize(cvImage, cvImage, cv::Size(), scaleX, scaleY);

                cv::GaussianBlur(cvImage, cvImage, cv::Size(13,13), 0);

                // cv::Mat xgrad, ygrad;
                // cv::Sobel(cvImage, xgrad, CV_16S, 1, 0, 3);
                // cv::Sobel(cvImage, ygrad, CV_16S, 0, 1, 3);
                // cv::convertScaleAbs(xgrad, xgrad);
                // cv::convertScaleAbs(ygrad, ygrad);
                // cv::addWeighted(xgrad, 0.5, ygrad, 0.5, 0, cvEdge);

                cv::Canny(cvImage, cvEdge, 50, 100);
                contourEdgeFilter(cvEdge, cvEdge, 30, 100);
                cv::resize(cvEdge, edgePair[j], cv::Size(imgWidth, imgHeight));

                cv::distanceTransform(255 - cvEdge, cvTrac, CV_DIST_L2, 5);
                cvTrac = cvTrac * 10;
                cvTrac.convertTo(cvTrac, CV_8UC1);
                cvTrac = 255 - cvTrac;
                cv::resize(cvTrac, cvTrac, cv::Size(imgWidth, imgHeight));
                cvTrac.copyTo(tracPair[j]);

                // cv::threshold(cvTrac, cvMask, outlierDistLim, 255, cv::THRESH_BINARY);
                // mask[j].resize(cvMask.cols, cvMask.rows);
                // cv::cv2eigen(cvMask, *(Image<unsigned char>::Base*)(mask+j));
            }

            omp_set_lock(&writelock);
            std::cout << i << " edge done" << std::endl;
            omp_unset_lock(&writelock);
        }

        if (!boost::filesystem::exists(matchDir+"/image_"+std::to_string(i*2)+".feat") ||
            !regions[0]->Load(matchDir+"/image_"+std::to_string(i*2)+".feat", matchDir+"/image_"+std::to_string(i*2)+".desc") ||
            !regions[1]->Load(matchDir+"/image_"+std::to_string(i*2+1)+".feat", matchDir+"/image_"+std::to_string(i*2+1)+".desc"))
        {
            omp_set_lock(&writelock);
            std::cout << i << " start detect" << std::endl;
            omp_unset_lock(&writelock);

            // ------------------------------------
            // 2. init image descriptor
            // ------------------------------------
            using namespace openMVG::features;
            std::unique_ptr<Image_describer> image_describer;

            if (featMethod == "mldb") {
                image_describer = AKAZE_Image_describer::create(AKAZE_Image_describer::Params(AKAZE::Params(), AKAZE_MLDB));
            }
            else if (featMethod == "surf") {
                image_describer = AKAZE_Image_describer::create(AKAZE_Image_describer::Params(AKAZE::Params(), AKAZE_MSURF));
            }
            else {
                SIFT_Anatomy_Image_describer::Params params;
                if(featDensity == "dense") {
                    params = SIFT_Anatomy_Image_describer::Params(0,6,3,10.0f,0.01f,true); // HIGH
                } else {
                    params = SIFT_Anatomy_Image_describer::Params(0,6,3,10.0f,0.04f,true); // NORMAL
                }
                image_describer.reset(new SIFT_Anatomy_Image_describer(params));
            }

            // ------------------------------------
            // 3. detect region
            // ------------------------------------
            for (int j=0; j<2; j++) {
                const int idx = i*2 + j;
                Image<unsigned char> &image = imagePair[j];

                regions[j] = image_describer->Describe(image, nullptr);
                if (!regions[j]) {
                    std::cout << "Invalid region: " << idx << std::endl;
                    continue;
                }

                if(!image_describer->Save(regions[j].get(), 
                    matchDir+"/image_"+std::to_string(idx)+".feat", 
                    matchDir+"/image_"+std::to_string(idx)+".desc")) {
                    std::cerr << "Cannot save regions for images: " << idx << std::endl
                            << "Stopping feature extraction." << std::endl;
                    continue;
                }
            }
        }
        else {
            omp_set_lock(&writelock);
            std::cout << i << " use cached" << std::endl;
            omp_unset_lock(&writelock);
        }

        omp_set_lock(&writelock);
        std::cout << i << " start match" << std::endl;
        omp_unset_lock(&writelock);

        // ------------------------------------
        // 4. compute corresponding points
        // ------------------------------------
        using KernelType = ACKernelAdaptor<
            openMVG::fundamental::kernel::SevenPointSolver,
            openMVG::fundamental::kernel::SymmetricEpipolarDistanceError,
            UnnormalizerT,
            Mat3 >;

        const std::unique_ptr<RegionsMatcher> matcher =
            RegionMatcherFactory(CASCADE_HASHING_L2, *regions[0].get());
        IndMatches vec_PutativeMatches;
        matcher->MatchDistanceRatio(0.8, *regions[1].get(), vec_PutativeMatches);

        omp_set_lock(&writelock);
        std::cout << i << " raw match " << vec_PutativeMatches.size() << std::endl;
        omp_unset_lock(&writelock);

        Mat xL(2, vec_PutativeMatches.size());
        Mat xR(2, vec_PutativeMatches.size());
        const auto &featsL = regions[0]->GetRegionsPositions();
        const auto &featsR = regions[1]->GetRegionsPositions();
        for (size_t k=0; k<vec_PutativeMatches.size(); ++k)  {
            const auto & imaL = featsL[vec_PutativeMatches[k].i_];
            const auto & imaR = featsR[vec_PutativeMatches[k].j_];
            xL.col(k) = imaL.coords().cast<double>();
            xR.col(k) = imaR.coords().cast<double>();
        }

        std::vector<uint32_t> vec_inliers;
        KernelType kernel(
            xL, imgWidth, imgHeight,
            xR, imgWidth, imgHeight,
            true); // configure as point to line error model.

        Mat3 F;
        // const std::pair<double,double> ACRansacOut = ACRANSAC(
        //     kernel, vec_inliers, 1024, &F,
        //     Square(10.0), // Upper bound of authorized threshold
        //     false);
        const std::pair<double,double> ACRansacOut = ACRANSAC(
            kernel, vec_inliers, 1024, &F);

        omp_set_lock(&writelock);
        std::cout << i << " ACRANSAC " << vec_inliers.size() << std::endl;
        omp_unset_lock(&writelock);

        double ratio = static_cast<double>(vec_inliers.size()) / vec_PutativeMatches.size();
        if (ratio > 0.3) {
            IndMatches vec_GeometricMatches;
            for(int k=0; k<vec_inliers.size(); k++) {
                auto match = vec_PutativeMatches[vec_inliers[k]];

                if (edgeCheck) {
                    Eigen::Vector2i uv1, uv2;
                    uv1 = featsL[match.i_].coords().cast<int>();
                    int trac1 = tracPair[0].at<unsigned char>(uv1(1), uv1(0));
                    if (trac1 >= outlierDistLim) {
                        vec_GeometricMatches.push_back(std::move(match));
                    }
                }
                else {
                    vec_GeometricMatches.push_back(std::move(match));
                }
            }

            omp_set_lock(&writelock);
            std::cout << i << " geometric " << vec_GeometricMatches.size() << std::endl;
            omp_unset_lock(&writelock);

            Pair pairIdx(i*2, i*2+1);
            omp_set_lock(&writelock);
            pairwiseMatches.insert(std::make_pair(pairIdx, std::move(vec_GeometricMatches)));
            omp_unset_lock(&writelock);

            const IndMatches &matches = pairwiseMatches.at(pairIdx);

            if (debugOutput && i%10==0) {
                cv::Mat cvImageI, cvImageJ;
                cv::eigen2cv(imagePair[0].GetMat(), cvImageI);
                cv::eigen2cv(imagePair[1].GetMat(), cvImageJ);

                cv::Mat jointImg;
                cv::hconcat(cvImageI, cvImageJ, jointImg);
                cv::cvtColor(jointImg, jointImg, cv::COLOR_GRAY2BGR);

                if (edgeCheck) {
                    cv::Mat jointMask, jointEdge, edgeLayer;
                    cv::hconcat(tracPair[0], tracPair[1], jointMask);
                    cv::hconcat(edgePair[0], edgePair[1], jointEdge);
                    cv::Mat blank_ch = cv::Mat::zeros(jointMask.size(), CV_8UC1);
                    std::vector<cv::Mat> channels = { jointMask, jointEdge, jointEdge };
                    cv::merge(channels, edgeLayer);
                    cv::addWeighted(jointImg, 1, edgeLayer, 1, 0, jointImg);
                }

                for (const auto &feature : featsL) {
                    cv::Mat pos;
                    cv::eigen2cv(feature.coords(), pos);
                    cv::circle(jointImg, cv::Point(pos), 3, cv::Scalar(0,255,0));
                }
                for (const auto &feature : featsR) {
                    cv::Mat pos;
                    cv::eigen2cv(feature.coords(), pos);
                    pos.at<float>(0) += imgWidth;
                    cv::circle(jointImg, cv::Point(pos), 3, cv::Scalar(0,255,0));
                }
                for (int k=0; k<matches.size(); k++) {
                    const auto &match = matches[k];
                    cv::Mat pos1, pos2;
                    cv::eigen2cv(featsL[match.i_].coords(), pos1);
                    cv::eigen2cv(featsR[match.j_].coords(), pos2);
                    pos2.at<float>(0) += imgWidth;
                    cv::Scalar col(0,0,255);
                    cv::line(jointImg, cv::Point(pos1), cv::Point(pos2), col);
                    cv::circle(jointImg, cv::Point(pos1), 3, col, 5);
                    cv::circle(jointImg, cv::Point(pos2), 3, col, 5);
                }

                cv::imwrite(matchDir+"/debug/"+std::to_string(i*2)+"-"+std::to_string(i*2+1)+".png", jointImg);
            
                if (edgeCheck) {
                    cv::Mat tracI = tracPair[0];

                    for (int k=0; k<matches.size(); k++) {
                        const auto &match = matches[k];
                        cv::Mat pos1;
                        cv::eigen2cv(featsL[match.i_].coords(), pos1);
                        cv::Scalar col(255, 255, 255);
                        cv::circle(tracI, cv::Point(pos1), 3, col, 5);
                    }

                    cv::imwrite(matchDir+"/debug/trac_"+std::to_string(i*2)+".png", tracI);
                }
            }
        } else {
            std::cerr << "Match inlier ratio < 0.3 at stero pair " << i*2 << "-" << i*2+1 << std::endl;
        }

        omp_set_lock(&writelock);
        std::cout << "Processed " << ++count << " / " << nImagePair << ", pair idx: " << i << std::endl;
        omp_unset_lock(&writelock);
    }
    std::cout << "Total time: " << timer.elapsed() << " seconds." << std::endl;

    // output matches
    std::string match_filename = edgeCheck ? matchDir+"/matches_edge.bin" : matchDir+"/matches.bin";
    if (!Save(pairwiseMatches, match_filename)) {
        std::cerr << "Cannot save computed matches in: " << match_filename << std::endl;
        return EXIT_FAILURE;
    }

    return 0;
}
