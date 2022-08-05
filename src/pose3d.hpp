#ifndef __POSE3D_HPP__
#define __POSE3D_HPP__

#include <fstream>

#include <Eigen/Dense>

typedef Eigen::Matrix3d Mat3;
typedef Eigen::Matrix4d Mat4;
typedef Eigen::MatrixXd Mat;
typedef Eigen::Vector3d Vec3;
typedef Eigen::Vector2d Vec2;
typedef Eigen::Vector4d Vec4;
typedef Eigen::Quaterniond Quaternion;
typedef std::pair<uint32_t, uint32_t> Pair;

struct Pose3D {
    Quaternion rotation_;
    Vec3 translation_;

    Pose3D() {
        rotation_ = Quaternion(1.0,0.0,0.0,0.0);
        translation_ = Vec3(0.0,0.0,0.0);
    }
    Pose3D(const Quaternion &rot, const Vec3 &t) {
        rotation_ = rot.normalized();
        translation_ = t;
    }

    Pose3D(const Mat4 &T) {
        const Quaternion q(T.block<3,3>(0,0));
        const Vec3 t(T.block<3,1>(0,3));
        rotation_ = q.normalized();
        translation_ = t;
    }

    const Quaternion& rotation() const {
        return rotation_;
    }

    const Vec3& translation() const {
        return translation_;
    }

    Pose3D inverse() const {
        const Quaternion q = rotation_.conjugate();
        return Pose3D(q, -(q*translation_));
    } 


    Mat4 toMat4() {
        return (Mat4() << rotation_.normalized().toRotationMatrix(), translation_,
                          0.0,0.0,0.0,1.0).finished();
    }
};

Pose3D operator*(const Pose3D& lhs,
                 const Pose3D& rhs) {
    return Pose3D(
      (lhs.rotation() * rhs.rotation()).normalized(),
      lhs.rotation() * rhs.translation() + lhs.translation());
}

Vec3 operator*(const Pose3D& pose,
               const Vec3& point) {
    return pose.rotation() * point + pose.translation();
}

std::ostream& operator<<(std::ostream& os,
                         const Pose3D& pose) {
  os << pose.rotation().w() << ", "
     << pose.rotation().x() << ", "
     << pose.rotation().y() << ", "
     << pose.rotation().z() << ", "
     << pose.translation().x() << ", "
     << pose.translation().y() << ", "
     << pose.translation().z() << ", "
     << std::endl;
  return os;
}

#endif
