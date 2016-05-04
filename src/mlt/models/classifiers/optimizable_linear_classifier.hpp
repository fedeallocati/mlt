#ifndef MLT_MODELS_OPTIMIZABLE_LINEAR_CLASSIFIER_HPP
#define MLT_MODELS_OPTIMIZABLE_LINEAR_CLASSIFIER_HPP

#include <Eigen/Core>

#include "../optimizable_linear_model.hpp"
#include "linear_classifier.hpp"

namespace mlt {
namespace models {
namespace classifiers {
	template <class Loss, class Optimizer>
	class OptimizableLinearClassifier : public OptimizableLinearModel<LinearClassifier<OptimizableLinearClassifier<Loss, Optimizer>>, Loss, Optimizer> {
	public:
		template <class L, class O, class = std::enable_if<std::is_same<std::decay_t<L>, Loss>::value && std::is_convertible<std::decay_t<O>, Optimizer>::value>>
		explicit OptimizableLinearClassifier(L&& loss, O&& optimizer, double regularization,
			bool fit_intercept = true) : OptimizableLinearModel(std::forward<L>(loss), std::forward<O>(optimizer), regularization, fit_intercept) {}

		OptimizableLinearClassifier& fit(const Eigen::Ref<const Eigen::MatrixXd>& input, const Eigen::Ref<const Eigen::VectorXi>& classes, bool cold_start = true) {
			this->_fit(input, this->_convert_to_classes_matrix(classes).cast<double>(), cold_start);
			return *this;
		}
	};
}
}
}
#endif