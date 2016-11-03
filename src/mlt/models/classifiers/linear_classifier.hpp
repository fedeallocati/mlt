#ifndef MLT_MODELS_CLASSIFIERS_LINEAR_CLASSIFIER_HPP
#define MLT_MODELS_CLASSIFIERS_LINEAR_CLASSIFIER_HPP

#include "../linear_model.hpp"
#include "classifier.hpp"
#include "../../utils/eigen.hpp"

namespace mlt {
namespace models {
namespace classifiers {
	template <class ConcreteType>
	class LinearClassifier : public LinearModel<Classifier<ConcreteType>> {
	public:
		Result predict(Features input) const {
			assert(_fitted);

			auto scores = _apply_linear_transformation(input);

			auto result = Result(input.cols());
			for (size_t col = 0; col < scores.cols(); col++) {
				result(col) = max_row(scores.col(col));
			}

			return result;
		}

		inline auto num_classes() { assert(_fitted); return coefficients().rows(); }

	protected:
		LinearClassifier(bool fit_intercept) : LinearModel(fit_intercept) {}
		LinearClassifier(const LinearClassifier&) = default;
		LinearClassifier(LinearClassifier&&) = default;
		LinearClassifier& operator=(const LinearClassifier&) = default;
		~LinearClassifier() = default;
	};
}
}
}
#endif