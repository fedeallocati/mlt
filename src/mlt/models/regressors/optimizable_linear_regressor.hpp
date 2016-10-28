#ifndef MLT_MODELS_REGRESSORS_OPTIMIZABLE_LINEAR_REGRESSOR_HPP
#define MLT_MODELS_REGRESSORS_OPTIMIZABLE_LINEAR_REGRESSOR_HPP

#include <type_traits>

#include <Eigen/Core>

#include "../optimizable_linear_model.hpp"
#include "linear_regressor.hpp"

namespace mlt {
namespace models {
namespace regressors {
	template <class Loss, class Optimizer>
	class OptimizableLinearRegressor : public OptimizableLinearModel<LinearRegressor<OptimizableLinearRegressor<Loss, Optimizer>>, Loss, Optimizer> {
	public:
		template <class L, class O, class = enable_if<is_same<decay_t<L>, Loss>::value && is_convertible<decay_t<O>, Optimizer>::value>>
		explicit OptimizableLinearRegressor(L&& loss, O&& optimizer, double regularization,
			bool fit_intercept = true) : OptimizableLinearModel(forward<L>(loss), forward<O>(optimizer), regularization, fit_intercept) {}
	};
}
}
}
#endif