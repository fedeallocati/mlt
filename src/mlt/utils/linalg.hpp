#ifndef MLT_UTILS_LINALNG_HPP
#define MLT_UTILS_LINALNG_HPP

#include <Eigen/Core>
#include <Eigen/SVD>

namespace mlt {
namespace utils {
namespace linalg {
	// Moore-Penrose pseudoinverse
	template <typename Derived>
	Eigen::Matrix<typename Derived::Scalar, Derived::RowsAtCompileTime, Derived::ColsAtCompileTime>
	pseudo_inverse(const Eigen::MatrixBase<Derived>& x) {
		Eigen::JacobiSVD<Eigen::Matrix<typename Derived::Scalar, Derived::RowsAtCompileTime, Derived::ColsAtCompileTime>> 
			svd(x, Eigen::ComputeThinU | Eigen::ComputeThinV);

		auto tolerance = std::numeric_limits<typename Derived::Scalar>::epsilon() * std::max(x.rows(), x.cols()) * svd.singularValues().maxCoeff();
		
		return svd.matrixV() * svd.singularValues().unaryExpr([=](double s) { return (s < tolerance) ? 0 : 1 / s; }).asDiagonal() * svd.matrixU().transpose();
	}
}
}
}
#endif