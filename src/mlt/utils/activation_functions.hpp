#ifndef MLT_UTILS_ACTIVATION_FUNCTIONS_HPP
#define MLT_UTILS_ACTIVATION_FUNCTIONS_HPP

#include <Eigen/Core>

#include "../defs.hpp"

namespace mlt {
namespace utils {
namespace activation_functions {
	class SigmoidActivation {
	public:
		auto compute(MatrixXdRef x) const {
			return x.unaryExpr([](double z) { double gz = 1.0 / (1.0 + std::exp(-z)); gz = gz < 1 ? gz : 0.9999999999; return gz; }).eval();
		}

		auto gradient(MatrixXdRef x) const {
			return x.unaryExpr([](double z) { double gz = 1.0 / (1.0 + std::exp(-z)); gz = gz < 1 ? gz : 0.9999999999; return gz * (1 - gz); }).eval();
		}
	};
}
}
}
#endif