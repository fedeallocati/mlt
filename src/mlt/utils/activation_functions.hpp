#ifndef MLT_UTILS_ACTIVATION_FUNCTIONS_HPP
#define MLT_UTILS_ACTIVATION_FUNCTIONS_HPP

#include <Eigen/Core>

namespace mlt {
namespace utils {
namespace activation_functions {
	class SigmoidActivation {
	public:
		Eigen::MatrixXd compute(const Eigen::Ref<const Eigen::MatrixXd>& x) const {
			return x.unaryExpr([](double z) { return 1.0 / (1.0 + exp(-z)); });
		}

		Eigen::MatrixXd gradient(const Eigen::Ref<const Eigen::MatrixXd>& x) const {
			return x.unaryExpr([](double z) { double gz = 1.0 / (1.0 + exp(-z)); return gz * (1 - gz); });
		}
	};
}
}
}
#endif