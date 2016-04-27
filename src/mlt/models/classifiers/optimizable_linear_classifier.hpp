#ifndef MLT_MODELS_OPTIMIZABLE_LINEAR_CLASSIFIER_HPP
#define MLT_MODELS_OPTIMIZABLE_LINEAR_CLASSIFIER_HPP

#include <Eigen/Core>

#include "../linear_model.hpp"
#include "../optimizable_linear_model.hpp"

namespace mlt {
namespace models {
namespace classifiers {
	template <class Loss, class Optimizer>
	class OptimizableLinearClassifier : public OptimizableLinearModel<OptimizableLinearClassifier<Loss, Optimizer>, LinearModel, Loss, Optimizer> {
	public:
		using OptimizableLinearModel<OptimizableLinearClassifier<Loss, Optimizer>, LinearModel, Loss, Optimizer>::OptimizableLinearModel;
		/*template <typename L, typename O, class = std::enable_if<std::is_same<std::decay_t<L>, Loss>::value && std::is_convertible<std::decay_t<O>, Optimizer>::value>>
		explicit OptimizableLinearClassifier(L&& loss, O&& optimizer, double regularization,
			bool fit_intercept = true) : OptimizableLinearModel(std::forward<L>(loss), std::forward<O>(optimizer), regularization, fit_intercept) {}*/

		OptimizableLinearClassifier& fit(const Eigen::Ref<const Eigen::MatrixXd>& input, const Eigen::Ref<const Eigen::MatrixXi>& classes, bool cold_start = true) {
			return this->fit(input, classes.cast<double>(), cold_start);
		}

		using OptimizableLinearModel<OptimizableLinearClassifier<Loss, Optimizer>, LinearModel, Loss, Optimizer>::fit;
	};
}
}
}
#endif