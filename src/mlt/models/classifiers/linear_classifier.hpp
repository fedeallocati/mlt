#ifndef MLT_MODELS_CLASSIFIERS_LINEAR_CLASSIFIER_HPP
#define MLT_MODELS_CLASSIFIERS_LINEAR_CLASSIFIER_HPP

#include <Eigen/Core>

#include "../linear_model.hpp"
#include "classifier_mixin.hpp"

namespace mlt {
namespace models {
namespace classifiers {
	template <class Classifier>
	class LinearClassifier : public LinearModel, public ClassifierMixin<Classifier> {
	public:
		Eigen::VectorXi classify(const Eigen::Ref<const Eigen::MatrixXd>& input) const {
			assert(_fitted);

			auto scores = this->_apply_linear_transformation(input);

			auto result = Eigen::VectorXi(input.cols());
			for (size_t col = 0; col < scores.cols(); col++) {
				int max_row;
				scores.col(col).maxCoeff(&max_row);
				result(col) = max_row;
			}

			return result;
		}

	protected:
		explicit LinearClassifier(bool fit_intercept) : LinearModel(fit_intercept) {}
	};
}
}
}
#endif