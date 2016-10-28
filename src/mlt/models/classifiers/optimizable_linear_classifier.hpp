#ifndef MLT_MODELS_CLASSIFIERS_OPTIMIZABLE_LINEAR_CLASSIFIER_HPP
#define MLT_MODELS_CLASSIFIERS_OPTIMIZABLE_LINEAR_CLASSIFIER_HPP

#include <type_traits>

#include <Eigen/Core>

#include "../optimizable_linear_model.hpp"
#include "linear_classifier.hpp"

namespace mlt {
namespace models {
namespace classifiers {
	template <class Loss, class Optimizer>
	class OptimizableLinearClassifier : public OptimizableLinearModel<LinearClassifier<OptimizableLinearClassifier<Loss, Optimizer>>, Loss, Optimizer> {
	public:
		template <class L, class O, class = enable_if<is_same<decay_t<L>, Loss>::value && is_convertible<decay_t<O>, Optimizer>::value>>
		explicit OptimizableLinearClassifier(L&& loss, O&& optimizer, double regularization,
			bool fit_intercept = true) : OptimizableLinearModel(forward<L>(loss), forward<O>(optimizer), regularization, fit_intercept) {}
	};
}
}
}
#endif