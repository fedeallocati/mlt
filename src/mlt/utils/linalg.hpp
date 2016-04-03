#ifndef MLT_UTILS_LINALNG_HPP
#define MLT_UTILS_LINALNG_HPP

#include <Eigen/Core>
#include <Eigen/SVD>

namespace mlt {
namespace utils {
namespace linalg {
	// TARGET TYPE: "Eigen::GeneralProduct<Eigen::DiagonalProduct<Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::DiagonalWrapper<const Eigen::Matrix<double, -1, 1, 0, -1, 1>>, 2>, Eigen::Transpose<const Eigen::Matrix<double, -1, -1, 0, -1, -1>>, 5>" 
	// "Eigen::GeneralProduct<Eigen::DiagonalProduct<Eigen::CwiseBinaryOp<Eigen::internal::scalar_sum_op<double>, const Eigen::GeneralProduct<Eigen::Transpose<Eigen::Matrix<double, -1, -1, 0, -1, -1>>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, 5>, const Eigen::CwiseUnaryOp<Eigen::internal::scalar_multiple_op<double>, const Eigen::CwiseNullaryOp<Eigen::internal::scalar_identity_op<double>, Eigen::Matrix<double, -1, -1, 0, -1, -1>>>>, Eigen::DiagonalWrapper<const Eigen::Matrix<double, -1, 1, 0, -1, 1>>, 2>, Eigen::Transpose<const Eigen::CwiseBinaryOp<Eigen::internal::scalar_sum_op<double>, const Eigen::GeneralProduct<Eigen::Transpose<Eigen::Matrix<double, -1, -1, 0, -1, -1>>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, 5>, const Eigen::CwiseUnaryOp<Eigen::internal::scalar_multiple_op<double>, const Eigen::CwiseNullaryOp<Eigen::internal::scalar_identity_op<double>, Eigen::Matrix<double, -1, -1, 0, -1, -1>>>>>, 5>"
	// WRITTEN TYPE: typename Eigen::GeneralProduct<Eigen::DiagonalProduct<Derived, Eigen::DiagonalWrapper<const Eigen::Matrix<typename Derived::Scalar, -1, 1, 0>>, 2>, Eigen::Transpose<const Derived>, 5>::Type
	// Moore-Penrose pseudoinverse
	template <typename Derived>
	Eigen::Matrix<typename Derived::Scalar, -1, -1>
	pseudo_inverse(const Eigen::MatrixBase<Derived>& x) {
		auto svd = x.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV);

		auto tolerance = std::numeric_limits<typename Derived::Scalar>::epsilon() * std::max(x.rows(), x.cols()) * svd.singularValues().maxCoeff();
		
		return svd.matrixV() * svd.singularValues().unaryExpr([=](typename Derived::Scalar s) { return (s < tolerance) ? 0 : 1 / s; }).eval().asDiagonal() * svd.matrixU().transpose();
	}
}
}
}
#endif