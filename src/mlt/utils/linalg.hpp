#ifndef MLT_UTILS_LINALNG_HPP
#define MLT_UTILS_LINALNG_HPP

#include <Eigen/Core>
#include <Eigen/SVD>

namespace mlt {
namespace utils {
namespace linalg {

	// Moore-Penrose pseudoinverse
	inline Eigen::MatrixXd
	pseudo_inverse(const Eigen::MatrixXd& x) {
		auto svd = x.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV);

		auto tolerance = std::numeric_limits<double>::epsilon() * std::max(x.rows(), x.cols()) * svd.singularValues().maxCoeff();
		
		return svd.matrixV() * svd.singularValues().unaryExpr([=](double s) { return (s < tolerance) ? 0 : 1 / s; }).eval().asDiagonal() * svd.matrixU().transpose();
	}

	inline Eigen::MatrixXd
	covariance(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y)
	{
		assert(x.rows() == y.rows());
		const auto num_observations = static_cast<double>(x.rows());
		const Eigen::RowVectorXd x_mean = x.colwise().sum() / num_observations;
		const Eigen::RowVectorXd y_mean = y.colwise().sum() / num_observations;
 
		return (x.rowwise() - x_mean).transpose() * (y.rowwise() - y_mean) / num_observations;
	}

	inline Eigen::MatrixXd
	covariance(const Eigen::MatrixXd& x)
	{
		return covariance(x, x);
	}
}
}
}
#endif