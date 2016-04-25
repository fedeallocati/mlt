#ifndef MLT_UTILS_LINEAR_ALGEBRA_HPP
#define MLT_UTILS_LINEAR_ALGEBRA_HPP

#include <Eigen/Core>
#include <Eigen/SVD>

namespace mlt {
namespace utils {
namespace linear_algebra {
	// Moore-Penrose pseudoinverse
	inline Eigen::MatrixXd pseudo_inverse(const Eigen::MatrixXd& x) {
		auto svd = x.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV);

		auto tolerance = std::numeric_limits<double>::epsilon() * std::max(x.rows(), x.cols()) * svd.singularValues().maxCoeff();
		
		return svd.matrixV() * svd.singularValues().unaryExpr([=](double s) { return (s < tolerance) ? 0 : 1 / s; }).eval().asDiagonal() * svd.matrixU().transpose();
	}

	inline Eigen::MatrixXd covariance(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y)
	{
		assert(x.cols() == y.cols());
		const auto num_observations = static_cast<double>(x.cols());
		return (x.colwise() - (x.rowwise().sum() / num_observations)) * (y.colwise() - (y.rowwise().sum() / num_observations)).transpose() / num_observations;
	}

	inline Eigen::MatrixXd covariance(const Eigen::MatrixXd& x)
	{
		return covariance(x, x);
	}
}
}
}
#endif