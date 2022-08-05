#include <iostream>

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <ceres/cubic_interpolation.h>

#include <Eigen/Core>

#include <opencv2/opencv.hpp>

#include <boost/format.hpp>

using namespace std;
using namespace Eigen;


struct GetPixelGrayValue {
    GetPixelGrayValue(const Mat3 K,
                      const Vec3 pos,
                      const double target,
                      const int rows, const int cols,
                      const vector<float> &vec_pixel_gray_values)
    {
        K_ = K;
        pos_ = pos;
        target_ = target;
        rows_ = rows; cols_ = cols;
        
        grid2d.reset(new ceres::Grid2D<float>(
            &(vec_pixel_gray_values[0]), 0, rows_, 0, cols_));
        get_pixel_gray_val.reset(
            new ceres::BiCubicInterpolator<ceres::Grid2D<float> >(*grid2d));
    }
    
    template <typename T>
    bool operator()(const T* const rot,
                    const T* const x,
                    const T* const y,
                    const T* const z,
                    T* residual) const
    {
        Eigen::Matrix<T, 3, 1> pt = Eigen::Quaternion<T>(rot[0], rot[1], rot[2], rot[3]) * pos_.cast<T>()
                                    + Eigen::Matrix<T, 3, 1>(*x, *y, *z);
        Eigen::Matrix<T, 3, 1> uv = K_.cast<T>() * pt;
        uv /= uv(2, 0);
                
        for (int i = 0; i < 9; i++)
        {
            int m = i % 3;
            int n = i / 3;
            T u, v, pixel_gray_val_out;
            u = uv(0, 0) + T(m - 1);
            v = uv(1, 0) + T(n - 1);
            get_pixel_gray_val->Evaluate(v, u, &pixel_gray_val_out);
            residual[i] = T(target_) - pixel_gray_val_out;
        }
        
        return true;
    }

    Mat3 K_;
    Vec3 pos_;
    double target_;
    int rows_, cols_;
    unique_ptr<ceres::Grid2D<float> > grid2d; 
    unique_ptr<ceres::BiCubicInterpolator<ceres::Grid2D<float> > > get_pixel_gray_val;
};

struct TransformationDelta {
    TransformationDelta(const Vec3 &initial,
                        const double weight)
    {
        initial_ = initial;
        weight_ = weight;
    }
    
    template <typename T>
    bool operator()(const T* const rot,
                    const T* const trans,
                    T* residual) const
    {
        T delta[3] = { trans[0]-T(initial_(0)), trans[1]-T(initial_(1)), trans[2]-T(initial_(2)) };
        residual[0] = T(weight_) * (delta[0]*delta[0] + delta[1]*delta[1] + delta[2]*delta[2]);
        
        return true;
    }

    Vec3 initial_;
    double weight_;
};


void directMethordBA( const Mat3 &K,
                      const Mat3 &R, const Vec3 &C, 
                      const vector<vector<Vec3>> &points,
                      const vector<vector<double>> &targets,
                      const vector<cv::Mat> &tractionMaps,
                      Mat3 &R_out, Vec3 &C_out,
                      const int max_iterations=30,
                      const string optimizeTarget="rt" ) {
    
    // cout << "  Preparing for optimization ... " << endl;

    Quaterniond q = Quaterniond(R).normalized();
    double rot[4] = { q.w(), q.x(), q.y(), q.z() };
    double trans[3] = { C(0), C(1), C(2) };
    
    int nImage = tractionMaps.size();
    vector<vector<float> > img_gray_values(nImage);
    for (int i = 0; i < nImage; i++) {
        int rows = tractionMaps[i].rows;
        int cols = tractionMaps[i].cols;
        for (int v = 0; v < rows; ++v)
            for (int u = 0; u < cols; ++u) 
                img_gray_values[i].push_back(tractionMaps[i].at<uint8_t>(v, u));
    }
        
    ceres::Problem problem;
    ceres::LocalParameterization *quaternion_parameterization =
        new ceres::QuaternionParameterization;
    ceres::LossFunction* loss_func(new ceres::HuberLoss(1.0));

    for (int i = 0; i < nImage; i++) {
        int nPoint = points[i].size();
        double inv_nPoint = 1.0 / nPoint;
        int rows = tractionMaps[i].rows;
        int cols = tractionMaps[i].cols;

        for (int j = 0; j < nPoint; j++) {
            problem.AddResidualBlock(
                new ceres::AutoDiffCostFunction<GetPixelGrayValue, 9, 4, 1, 1, 1> (
                    new GetPixelGrayValue ( K,
                                            points[i][j],
                                            targets[i][j],
                                            rows, cols,
                                            img_gray_values[i] )),
                loss_func,
                rot, trans, trans+1, trans+2
            );
        }

        // problem.AddResidualBlock(
        //     new ceres::AutoDiffCostFunction<TransformationDelta, 1, 4, 3> (
        //         new TransformationDelta ( C, 1000 )),
        //     loss_func,
        //     rot, trans
        // );
    }
    
    problem.SetParameterization(rot, quaternion_parameterization);

    const auto npos = std::string::npos;
    if (optimizeTarget.find('t') == npos && optimizeTarget.find('x') == npos) {
        problem.SetParameterBlockConstant(trans);
    }
    if (optimizeTarget.find('t') == npos && optimizeTarget.find('y') == npos) {
        problem.SetParameterBlockConstant(trans+1);
    }
    if (optimizeTarget.find('t') == npos && optimizeTarget.find('z') == npos) {
        problem.SetParameterBlockConstant(trans+2);
    }
    if (optimizeTarget.find('r') == npos) {
        problem.SetParameterBlockConstant(rot);
    }
    
    // cout << "  Solving ceres directBA ... " << endl;
    // cout << "    max interations: " << max_iterations << endl;
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::LinearSolverType::SPARSE_SCHUR;
    options.num_threads = 12;
    options.max_num_iterations = max_iterations;
    options.minimizer_progress_to_stdout = false;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    // cout << summary.FullReport() << "\n";

    R_out = Mat3(Quaterniond(rot[0], rot[1], rot[2], rot[3]).normalized());
    C_out = Vec3(trans[0], trans[1], trans[2]);
    
    return;
}


void directMethordBA( const Mat3 &K,
                      const Mat3 &R, const Vec3 &C, 
                      const vector<vector<Vec3>> &points,
                      const vector<cv::Mat> &tractionMap,
                      Mat3 &R_out, Vec3 &C_out,
                      const int max_iterations=30,
                      const string optimizeTarget="rt" ) {

    int nImage = tractionMap.size();
    vector<vector<double>> targets;
    for (int i = 0; i < nImage; i++) {
        vector<double> tmp(points[i].size(), 255.0);
        targets.push_back(move(tmp));
    }
    directMethordBA(K, R, C, points, targets, tractionMap, R_out, C_out, max_iterations, optimizeTarget);
}


void generateTractionMap(const cv::Mat &edgeImage, 
                         cv::Mat &tractionMap,
                         const int maxd = 100) {
    int rows, cols;
    rows = edgeImage.rows;
    cols = edgeImage.cols;

    tractionMap = cv::Mat::zeros(edgeImage.size(), CV_8UC1);

    const double alpha = 1.0/3.0, gamma = 0.98;

    #pragma omp parallel for
    for (int p1 = 0; p1 < rows * cols; p1++) {
        int i = p1 / cols;
        int j = p1 % cols;
        double maxval = 0;
        
        for (int d = 0; d < maxd; d++) {
            double coeff = pow(gamma, d);
            if (255 * coeff <= maxval) break;
            
            int x, y, lim;
            uint8_t maxpixel = 0;
            // up
            if (i - d >= 0) {
                x = i - d;
                lim = min(cols, j + d + 1);
                for (y = max(j - d, 0); y < lim; y++) {
                    maxpixel = max(maxpixel, edgeImage.at<uint8_t>(x, y));
                }
            }
            // down
            if (i + d < rows) {
                x = i + d;
                lim = min(cols, j + d + 1);
                for (y = max(j - d, 0); y < lim; y++) {
                    maxpixel = max(maxpixel, edgeImage.at<uint8_t>(x, y));
                }
            }
            // left
            if (j - d >= 0) {
                y = j - d;
                lim = min(rows, i + d + 1);
                for (x = max(i - d, 0); x < lim; x++) {
                    maxpixel = max(maxpixel, edgeImage.at<uint8_t>(x, y));
                }
            }
            // right
            if (j + d < cols) {
                y = j + d;
                lim = min(rows, i + d + 1);
                for (x = max(i - d, 0); x < lim; x++) {
                    maxpixel = max(maxpixel, edgeImage.at<uint8_t>(x, y));
                }
            }

            maxval = max(maxval, maxpixel * coeff);
        }

        double res = alpha * edgeImage.at<uint8_t>(i, j) + (1 - alpha) * maxval;
        if (res > 255) res = 255;
        tractionMap.at<uint8_t>(i, j) = res;
    }
}

vector<double> evalCost(const Mat3 &K,
                        const Mat3 &R, const Vec3 &C, 
                        const vector<vector<Vec3>> &points,
                        const vector<vector<double>> &targets,
                        const vector<cv::Mat> &tractionMaps) {

    Quaterniond q = Quaterniond(R).normalized();
    double rot[4] = { q.w(), q.x(), q.y(), q.z() };
    double trans[3] = { C(0), C(1), C(2) };
    
    int nImage = tractionMaps.size();
    vector<vector<float> > img_gray_values(nImage);
    for (int i = 0; i < nImage; i++) {
        int rows = tractionMaps[i].rows;
        int cols = tractionMaps[i].cols;
        for (int v = 0; v < rows; ++v)
            for (int u = 0; u < cols; ++u) 
                img_gray_values[i].push_back(tractionMaps[i].at<uint8_t>(v, u));
    }

    vector<double> costs(nImage);
    #pragma omp parallel for
    for (int i = 0; i < nImage; i++) {
        double totCost = 0;

        int nPoint = points[i].size();
        double inv_nPoint = 1.0 / nPoint;
        int rows = tractionMaps[i].rows;
        int cols = tractionMaps[i].cols;

        for (int j = 0; j < nPoint; j++) {
            GetPixelGrayValue costfunc( K,
                                        points[i][j],
                                        targets[i][j],
                                        rows, cols,
                                        img_gray_values[i]);
            
            double cost[9];
            costfunc(rot, trans, trans+1, trans+2, cost);
            totCost += cost[4];
        }

        costs[i] = totCost * inv_nPoint;
    }

    return costs;
}