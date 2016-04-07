#ifndef MLT_MODELS_CLASSIFIERS_LINEAR_CLASSIFIER_MODEL_HPP
#define MLT_MODELS_CLASSIFIERS_LINEAR_CLASSIFIER_MODEL_HPP

#include <Eigen/Core>

#include "../linear_model.hpp"

namespace mlt {
namespace models {
namespace classifiers {
	class LinearClassifierModel : public LinearModel {
	public:
		Eigen::MatrixXd predict(const Eigen::MatrixXd& input) const {
			assert(_fitted);

			return _score(input);
		}

	protected:
		explicit LinearClassifierModel(bool fit_intercept) : LinearModel(fit_intercept) {}
	};
}
}
}
#endif