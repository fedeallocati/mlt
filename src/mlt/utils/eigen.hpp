#ifndef MLT_UTILS_EIGEN_HPP
#define MLT_UTILS_EIGEN_HPP

#include <Eigen/Core>

namespace mlt {
namespace utils {
namespace eigen {
	inline Eigen::Map<const Eigen::MatrixXd> ravel(const Eigen::Ref<const Eigen::MatrixXd>& x)
	{
		return Eigen::Map<const Eigen::MatrixXd>(x.data(), x.size(), 1);
	}

	inline Eigen::Map<const Eigen::MatrixXd> unravel(const Eigen::Ref<const Eigen::MatrixXd>& x, size_t rows, size_t cols)
	{
		return Eigen::Map<const Eigen::MatrixXd>(x.data(), rows, cols);
	}
}
}
}
#endif