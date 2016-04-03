#ifndef MLT_UTILS_LINALNG_HPP
#define MLT_UTILS_LINALNG_HPP

#include <Eigen/Core>
#include <Eigen/SVD>

namespace mlt {
namespace utils {
namespace linalg {
	// Moore-Penrose pseudoinverse
	template <typename Derived>
	inline Eigen::MatrixXd pseudo_inverse(const Eigen::EigenBase<Derived>& x) {
		Eigen::JacobiSVD<Eigen::MatrixXd> svd(x, Eigen::ComputeThinU | Eigen::ComputeThinV);

		auto tolerance = std::numeric_limits<double>::epsilon() * std::max(x.rows(), x.cols()) * svd.singularValues().maxCoeff();
		
		return svd.matrixV() * svd.singularValues().unaryExpr([=](double s) { return (s < tolerance) ? 0 : 1 / s; }).asDiagonal() * svd.matrixU().transpose();
	}
}
}
}
#endif