#include <ros/ros.h>

#include "Eigen/Core"

#include "opencv2/core.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/core/eigen.hpp"

#include "file_io.hpp"


int main(int argc, char** argv) {

    ros::init(argc, argv, "image_preprocess_node");
    ros::NodeHandle n("~");

    std::string imageInDir, imageOutDir, calibDir;
    int sampleStep;
    n.param("image_input_dir",      imageInDir,     std::string(""));
    n.param("image_output_dir",     imageOutDir,    std::string(""));
    n.param("stereo_calib_dir",     calibDir,       std::string(""));
    n.param("sample_step",          sampleStep,     1);

    // List left and right images, using relative paths based on imageInDir
    std::vector<std::string> leftImageFileNames;
    std::vector<std::string> rightImageFileNames;
    std::vector<std::string> thermalImageFileNames;
    // getFilenamesByTypeAndDirPrefix(imageInDir, ".png", "left", leftImageFileNames);
    // getFilenamesByTypeAndDirPrefix(imageInDir, ".png", "right", rightImageFileNames);
    // getFilenamesByTypeAndDirPrefix(imageInDir, ".png", "thermal", thermalImageFileNames);
    readData(imageInDir+"/left/left_filelist.csv", leftImageFileNames);
    readData(imageInDir+"/right/right_filelist.csv", rightImageFileNames);
    readData(imageInDir+"/thermal/thermal_filelist.csv", thermalImageFileNames);

    // Create output folder
    createCleanFolder(imageOutDir);
    createCleanFolder(imageOutDir + "/left");
    createCleanFolder(imageOutDir + "/right");
    createCleanFolder(imageOutDir + "/thermal");

    if(leftImageFileNames.size() != rightImageFileNames.size()) {
        std::cerr << "Left and right images are not the same. Check folder: "
            << imageInDir << std::endl;
        return 1;
    } else if(leftImageFileNames.size() != thermalImageFileNames.size()) {
        std::cerr << "Rgb and thermal images are not the same. Check folder: "
            << imageInDir << std::endl;
        return 1;
    } else {
        std::cout << "Found " << leftImageFileNames.size() << " image groups. " << std::endl;
    }
    
    size_t nImagePair = leftImageFileNames.size();
    if(nImagePair == 0) {
        std::cerr << "No image files (.png or .jpg) are found in given folder. " << std::endl;
        return 1;
    }

    // ------------------------------------
    // stereo
    // ------------------------------------
    // Load a sample image to get image meta info
    cv::Mat img = cv::imread(leftImageFileNames[0]);
    // Load stereo calibration parameters
    Eigen::MatrixXd K1; 
    Eigen::MatrixXd K2; 
    Eigen::MatrixXd D1; 
    Eigen::MatrixXd D2; 
    Eigen::MatrixXd R;  
    Eigen::MatrixXd T;  
    readData(calibDir+"/stereo/CameraMatrixLeft.dat", 3, 3, ' ', K1);
    readData(calibDir+"/stereo/CameraMatrixRight.dat", 3, 3, ' ', K2);
    readData(calibDir+"/stereo/DistortionCoefficientLeft.dat", 1, 5, ' ', D1);
    readData(calibDir+"/stereo/DistortionCoefficientRight.dat", 1, 5, ' ', D2);
    readData(calibDir+"/stereo/T.dat", 3, 1, ' ', T);
    readData(calibDir+"/stereo/R.dat", 3, 3, ' ', R);

    std::cout << "K1: " << K1 << std::endl;
    std::cout << "K2: " << K2 << std::endl;
    std::cout << "D1: " << D1 << std::endl;
    std::cout << "D2: " << D2 << std::endl;
    std::cout << "R: " << R  << std::endl;
    std::cout << "T: " << T  << std::endl;

    cv::Mat cameraMatrix1, cameraMatrix2;
    cv::Mat distCoeffs1, distCoeffs2;
    cv::Mat Rotation, Translation;
    cv::eigen2cv(K1, cameraMatrix1);
    cv::eigen2cv(K2, cameraMatrix2);
    cv::eigen2cv(D1, distCoeffs1);
    cv::eigen2cv(D2, distCoeffs2);
    cv::eigen2cv(R, Rotation);
    cv::eigen2cv(T, Translation);

    cv::Mat R1, R2, P1, P2, Q, newK1, newK2;
    std::cout << img.size() << std::endl;
    cv::stereoRectify(cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, 
                      img.size(), Rotation, Translation, 
                      R1, R2, P1, P2, Q, 
                      CV_CALIB_ZERO_DISPARITY, 0);
    newK1 = cv::Mat(P1, cv::Rect(0,0,3,3));
    newK2 = cv::Mat(P2, cv::Rect(0,0,3,3));

    std::cout << "Rectification parameters: " << std::endl;
    std::cout << "R1: " << R1 << std::endl;
    std::cout << "R2: " << R2 << std::endl;
    std::cout << "P1: " << P1 << std::endl;
    std::cout << "P2: " << P2 << std::endl;

    cv::Mat map11, map12;
    cv::Mat map21, map22;
    cv::initUndistortRectifyMap(cameraMatrix1, distCoeffs1, R1, newK1, img.size(), CV_32FC1, map11, map12);
    cv::initUndistortRectifyMap(cameraMatrix2, distCoeffs2, R2, newK2, img.size(), CV_32FC1, map21, map22);

    // ------------------------------------
    // thermal
    // ------------------------------------
    img = cv::imread(thermalImageFileNames[0]);

    Eigen::MatrixXd K3; 
    Eigen::MatrixXd D3;
    readData(calibDir+"/thermal/CameraMatrix.dat", 3, 3, ' ', K3);
    readData(calibDir+"/thermal/DistortionCoefficient.dat", 1, 5, ' ', D3);

    cv::Mat cameraMatrix3;
    cv::Mat distCoeffs3;
    cv::eigen2cv(K3, cameraMatrix3);
    cv::eigen2cv(D3, distCoeffs3);

    cv::Mat newK3;
    newK3 = cv::getOptimalNewCameraMatrix(cameraMatrix3, distCoeffs3, img.size(), 0, img.size());


    size_t count = 0;
    std::vector<std::string> rectifiedLeftImageNames;
    std::vector<std::string> rectifiedRightImageNames;
    std::vector<std::string>rectifiedThermalImageNames;
    size_t nSelectedImagePair = (nImagePair-1)/sampleStep + 1;
    rectifiedLeftImageNames.resize(nSelectedImagePair);
    rectifiedRightImageNames.resize(nSelectedImagePair);
    rectifiedThermalImageNames.resize(nSelectedImagePair);

    std::cout << "Rectifiying every " << sampleStep << " image pairs." << std::endl;

    omp_lock_t writelock;
    omp_init_lock(&writelock);
    #pragma omp parallel for 
    for(size_t i=0; i<nSelectedImagePair; i++) {
        cv::Mat img1 = cv::imread(leftImageFileNames[i*sampleStep]);
        cv::Mat img2 = cv::imread(rightImageFileNames[i*sampleStep]);
        cv::Mat img3 = cv::imread(thermalImageFileNames[i*sampleStep], -1);

        cv::Mat rect1, rect2, rect3;
        cv::remap(img1, rect1, map11, map12, cv::INTER_LINEAR);
        cv::remap(img2, rect2, map21, map22, cv::INTER_LINEAR);
        cv::undistort(img3, rect3, cameraMatrix3, distCoeffs3, newK3);
        
        std::string leftImageFileName = leftImageFileNames[i*sampleStep].replace(0, imageInDir.length(), imageOutDir);
        std::string rightImageFileName = rightImageFileNames[i*sampleStep].replace(0, imageInDir.length(), imageOutDir);;
        std::string thermalImageFileName = thermalImageFileNames[i*sampleStep].replace(0, imageInDir.length(), imageOutDir);;
        cv::imwrite(leftImageFileName, rect1);
        cv::imwrite(rightImageFileName, rect2);
        cv::imwrite(thermalImageFileName, rect3);

        rectifiedLeftImageNames[i] = leftImageFileName;
        rectifiedRightImageNames[i] = rightImageFileName;
        rectifiedThermalImageNames[i] = thermalImageFileName;
        omp_set_lock(&writelock);
        std::cout << "Rectified group " << ++count << "/" << nImagePair/sampleStep << "\r" << std::flush;
        omp_unset_lock(&writelock);
    }
    std::cout << std::endl;

    writeData(imageOutDir+"/P1.dat", P1);
    writeData(imageOutDir+"/P2.dat", P2);
    writeData(imageOutDir+"/Q.dat", Q);
    writeData(imageOutDir+"/Kt.dat", newK3);
    writeData(imageOutDir+"/left_filenames.csv",  rectifiedLeftImageNames);
    writeData(imageOutDir+"/right_filenames.csv", rectifiedRightImageNames);
    writeData(imageOutDir+"/thermal_filenames.csv", rectifiedThermalImageNames);

    std::cout << "Rectification finished." << std::endl;
    std::cout << "Data saved to: " << imageOutDir << std::endl;

    return 0;
}
