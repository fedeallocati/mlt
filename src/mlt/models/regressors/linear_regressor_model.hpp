#ifndef MLT_MODELS_REGRESSORS_LINEAR_REGRESSOR_MODEL_HPP
#define MLT_MODELS_REGRESSORS_LINEAR_REGRESSOR_MODEL_HPP

#include <Eigen/Core>

#include "../linear_model.hpp"

namespace mlt {
namespace models {
namespace regressors {
	class LinearRegressorModel : public LinearModel {
	public:
		Eigen::MatrixXd predict(const Eigen::Ref<const Eigen::MatrixXd>& input) const {
			assert(_fitted);

			return _apply_linear_transformation(input);
		}

	protected:
		explicit LinearRegressorModel(bool fit_intercept) : LinearModel(fit_intercept) {}
	};
}
}
}
#endif