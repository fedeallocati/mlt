#ifndef MLT_MODELS_REGRESSORS_LINEAR_REGRESSOR_HPP
#define MLT_MODELS_REGRESSORS_LINEAR_REGRESSOR_HPP

#include "../linear_model.hpp"
#include "regressor.hpp"

namespace mlt {
namespace models {
namespace regressors {
	template <class ConcreteType>
	class LinearRegressor : public LinearModel<Regressor<ConcreteType>> {
	public:
		Result predict(Features input) const {
			assert(_fitted);

			return _apply_linear_transformation(input);
		}

	protected:
		LinearRegressor(bool fit_intercept) : LinearModel(fit_intercept) {}
		LinearRegressor(const LinearRegressor&) = default;
		LinearRegressor(LinearRegressor&&) = default;
		LinearRegressor& operator=(const LinearRegressor&) = default;
		~LinearRegressor() = default;
	};
}
}
}
#endif