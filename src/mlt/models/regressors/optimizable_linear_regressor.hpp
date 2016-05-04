#ifndef MLT_MODELS_OPTIMIZABLE_LINEAR_REGRESSOR_HPP
#define MLT_MODELS_OPTIMIZABLE_LINEAR_REGRESSOR_HPP

#include <Eigen/Core>

#include "../optimizable_linear_model.hpp"
#include "linear_regressor.hpp"

namespace mlt {
namespace models {
namespace regressors {
	template <class Loss, class Optimizer>
	class OptimizableLinearRegressor : public OptimizableLinearModel<LinearRegressor<OptimizableLinearRegressor<Loss, Optimizer>>, Loss, Optimizer> {
	public:
		template <class L, class O, class = std::enable_if<std::is_same<std::decay_t<L>, Loss>::value && std::is_convertible<std::decay_t<O>, Optimizer>::value>>
		explicit OptimizableLinearRegressor(L&& loss, O&& optimizer, double regularization,
			bool fit_intercept = true) : OptimizableLinearModel(std::forward<L>(loss), std::forward<O>(optimizer), regularization, fit_intercept) {}

		OptimizableLinearRegressor& fit(const Eigen::Ref<const Eigen::MatrixXd>& input, const Eigen::Ref<const Eigen::MatrixXd>& target, bool cold_start = true) {
			this->_fit(input, target, cold_start);
			return *this;
		}
	};
}
}
}
#endif