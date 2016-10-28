#ifndef MLT_UTILS_LINEAR_ALGEBRA_HPP
#define MLT_UTILS_LINEAR_ALGEBRA_HPP

#include <algorithm>
#include <limits>

#include <Eigen/Core>
#include <Eigen/SVD>

#include "../defs.hpp"

namespace mlt {
namespace utils {
namespace linear_algebra {
	// Moore-Penrose pseudoinverse
	inline auto pseudo_inverse(MatrixXdRef x) {
		auto svd = x.jacobiSvd(ComputeThinU | ComputeThinV);

		auto tolerance = numeric_limits<double>::epsilon() * max(x.rows(), x.cols()) * svd.singularValues().maxCoeff();
		
		return (svd.matrixV() * svd.singularValues().unaryExpr([=](double s) { return (s < tolerance) ? 0 : 1 / s; }).eval().asDiagonal() * svd.matrixU().transpose()).eval();
	}

	inline auto covariance(MatrixXdRef x, MatrixXdRef y) {
		assert(x.cols() == y.cols());
		const auto num_observations = static_cast<double>(x.cols());
		return ((x.colwise() - (x.rowwise().sum() / num_observations)) * (y.colwise() - (y.rowwise().sum() / num_observations)).transpose() / num_observations).eval();
	}

	inline auto linear_transformation(MatrixXdRef x, MatrixXdRef w) {
		return (w * x).eval();
	}

	inline auto linear_transformation(MatrixXdRef x, MatrixXdRef w, VectorXdRef b) {
		return ((w * x).colwise() + b).eval();
	}
}
}
}
#endif