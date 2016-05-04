#ifndef MLT_MODELS_REGRESSORS_LINEAR_REGRESSOR_HPP
#define MLT_MODELS_REGRESSORS_LINEAR_REGRESSOR_HPP

#include <Eigen/Core>

#include "../linear_model.hpp"
#include "regressor_mixin.hpp"

namespace mlt {
namespace models {
namespace regressors {
	template <class Regressor>
	class LinearRegressor : public LinearModel, public RegressorMixin<Regressor> {
	public:
		Eigen::MatrixXd predict(const Eigen::Ref<const Eigen::MatrixXd>& input) const {
			assert(_fitted);

			return _apply_linear_transformation(input);
		}

	protected:
		explicit LinearRegressor(bool fit_intercept) : LinearModel(fit_intercept) {}
	};
}
}
}
#endif