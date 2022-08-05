#include <iostream>

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <ceres/cubic_interpolation.h>

#include <Eigen/Core>

#include <opencv2/opencv.hpp>


struct PointToPlaneError{

    const Eigen::Vector3d& p_dst;
    const Eigen::Vector3d& p_src;
    const Eigen::Vector3d& p_nor;

    PointToPlaneError(const Eigen::Vector3d& src, const Eigen::Vector3d& dst, const Eigen::Vector3d& nor) :
    p_dst(dst), p_src(src), p_nor(nor)
    {
        // cout<<nor.dot(nor)<<endl;
    }

    // Factory to hide the construction of the CostFunction object from the client code.
    static ceres::CostFunction* Create(const Eigen::Vector3d& dst, const Eigen::Vector3d& src, const Eigen::Vector3d& nor) {
        return (new ceres::AutoDiffCostFunction<PointToPlaneError, 1, 4, 3>(new PointToPlaneError(dst, src, nor)));
    }

    template <typename T>
    bool operator()(const T* const camera_rotation, const T* const camera_translation, T* residuals) const {

        // Make sure the Eigen::Vector world point is using the ceres::Jet type as it's Scalar type
        Eigen::Matrix<T,3,1> src; src << T(p_src[0]), T(p_src[1]), T(p_src[2]);
        Eigen::Matrix<T,3,1> dst; dst << T(p_dst[0]), T(p_dst[1]), T(p_dst[2]);
        Eigen::Matrix<T,3,1> nor; nor << T(p_nor[0]), T(p_nor[1]), T(p_nor[2]);

        // Map the T* array to an Eigen Quaternion object (with appropriate Scalar type)
        Eigen::Quaternion<T> q = Eigen::Map<const Eigen::Quaternion<T> >(camera_rotation);
        // Map T* to Eigen Vector3 with correct Scalar type
        Eigen::Matrix<T,3,1> t = Eigen::Map<const Eigen::Matrix<T,3,1> >(camera_translation);
        // Rotate the point using Eigen rotations
        Eigen::Matrix<T,3,1> p = q * src;
        p += t;

        // The error is the difference between the predicted and observed position projected onto normal
        residuals[0] = (p - dst).dot(nor);

        return true;
    }
};

struct PointToPointError{

    const Eigen::Vector3d& p_dst;
    const Eigen::Vector3d& p_src;

    PointToPointError(const Eigen::Vector3d& src, const Eigen::Vector3d& dst) :
    p_dst(dst), p_src(src)
    {
    }

    // Factory to hide the construction of the CostFunction object from the client code.
    static ceres::CostFunction* Create(const Eigen::Vector3d& dst, const Eigen::Vector3d& src) {
        return (new ceres::AutoDiffCostFunction<PointToPointError, 3, 4, 3>(new PointToPointError(dst, src)));
    }

    template <typename T>
    bool operator()(const T* const camera_rotation, const T* const camera_translation, T* residuals) const {

        // Make sure the Eigen::Vector world point is using the ceres::Jet type as it's Scalar type
        Eigen::Matrix<T,3,1> src; src << T(p_src[0]), T(p_src[1]), T(p_src[2]);
        Eigen::Matrix<T,3,1> dst; dst << T(p_dst[0]), T(p_dst[1]), T(p_dst[2]);

        // Map the T* array to an Eigen Quaternion object (with appropriate Scalar type)
        Eigen::Quaternion<T> q = Eigen::Map<const Eigen::Quaternion<T> >(camera_rotation);
        // Map T* to Eigen Vector3 with correct Scalar type
        Eigen::Matrix<T,3,1> t = Eigen::Map<const Eigen::Matrix<T,3,1> >(camera_translation);
        // Rotate the point using Eigen rotations
        Eigen::Matrix<T,3,1> p = q * src;
        p += t;

        // The error is the difference between the predicted and observed position projected onto normal
        residuals[0] = p[0] - dst[0];
        residuals[1] = p[1] - dst[1];
        residuals[2] = p[2] - dst[2];

        return true;
    }
};

// void addPointToPlaneResidualBlocks( ceres::Problem &problem, 
//                                     Eigen::Quaterniond &q, Eigen::Vector3d &t,
//                                     const std::vector<Eigen::Vector3d> &src_points,
//                                     const std::vector<Eigen::Vector3d> &dst_points, 
//                                     const std::vector<Eigen::Vector3d> &dst_normals ) {
    
//     const int nPoint = src_points.size();
    
//     for(int i=0; i<nPoint; i++){
//         ceres::CostFunction* cost_function = 
//             PointToPlaneError::Create(src_points[i], dst_points[i], dst_normals[i]);

//         ceres::LossFunction* loss = new ceres::SoftLOneLoss(1.0);

//         problem.AddResidualBlock(cost_function, loss, q.coeffs().data(), t.data());
//     }
// }

// void addPointToPointResidualBlocks( ceres::Problem &problem, 
//                                     Eigen::Quaterniond &q, Eigen::Vector3d &t,
//                                     const std::vector<Eigen::Vector3d> &src_points,
//                                     const std::vector<Eigen::Vector3d> &dst_points ) {
    
//     const int nPoint = src_points.size();
    
//     for(int i=0; i<nPoint; i++){
//         ceres::CostFunction* cost_function = 
//             PointToPointError::Create(src_points[i], dst_points[i]);

//         ceres::LossFunction* loss = new ceres::SoftLOneLoss(1.0);

//         problem.AddResidualBlock(cost_function, loss, q.coeffs().data(), t.data());
//     }
// }

// void ceresSolve(ceres::Problem &problem,
//                 Eigen::Quaterniond &q, 
//                 Eigen::Vector3d &t) {

//     ceres::LocalParameterization *quaternion_parameterization =
//         new ceres::QuaternionParameterization;

//     problem.SetParameterization(q.coeffs().data(), quaternion_parameterization);

//     ceres::Solver::Options options;
//     options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
//     options.num_threads = 12;
//     options.max_num_iterations = 50;
//     ceres::Solver::Summary summary;
//     ceres::Solve(options, &problem, &summary);
// }

void ceresICPOptimizer( const Eigen::Matrix3d &R, const Eigen::Vector3d &C,
                        const std::vector<Eigen::Vector3d> &src_points,
                        const std::vector<Eigen::Vector3d> &dst_points, 
                        Eigen::Matrix3d &newR, Eigen::Vector3d &newC,
                        const std::string optimizeTarget="rt") {
    
    const int nPoint = src_points.size();

    ceres::Problem problem;
    ceres::LocalParameterization *quaternion_parameterization =
        new ceres::QuaternionParameterization;

    Eigen::Quaterniond q = Eigen::Quaterniond(R).normalized();
    Eigen::Vector3d t = C;

    for(int i=0; i<nPoint; i++){
        ceres::CostFunction* cost_function;
        cost_function = PointToPointError::Create(src_points[i], dst_points[i]);
        // if (dst_normals[i].isZero())
        //     cost_function = PointToPointError::Create(src_points[i], dst_points[i]);
        // else
        //     cost_function = PointToPlaneError::Create(src_points[i], dst_points[i], dst_normals[i]);

        ceres::LossFunction* loss = new ceres::SoftLOneLoss(1.0);

        problem.AddResidualBlock(cost_function, loss, q.coeffs().data(), t.data());
    }

    problem.SetParameterization(q.coeffs().data(), quaternion_parameterization);

    if (optimizeTarget.find('r') == std::string::npos) {
        problem.SetParameterBlockConstant(q.coeffs().data());
    }
    if (optimizeTarget.find('t') == std::string::npos) {
        problem.SetParameterBlockConstant(t.data());
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.num_threads = 12;
    options.max_num_iterations = 50;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    // std::cout << summary.FullReport() << std::endl;

    newR = Eigen::Matrix3d(q.normalized());
    newC = t;
}
