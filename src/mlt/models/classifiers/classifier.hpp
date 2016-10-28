#ifndef MLT_MODELS_CLASSIFIERS_CLASSIFIER_HPP
#define MLT_MODELS_CLASSIFIERS_CLASSIFIER_HPP

#include <Eigen/Core>

#include "../base.hpp"
#include "../../utils/eigen.hpp"

namespace mlt {
namespace models {
namespace classifiers {
	using namespace utils::eigen;

	template <class ConcreteType>
	class Classifier : public Predictor<ConcreteType, VectorXi> {
	public:
		inline auto score(Features input, Target target) const {
			return (_self().predict(input).array() == target.array()).cast<double>().sum() / static_cast<double>(input.cols());
		}

	protected:
		Classifier() = default;
		Classifier(const Classifier&) = default;
		Classifier(Classifier&&) = default;
		Classifier& operator=(const Classifier&) = default;
		~Classifier() = default;

		inline auto _to_target_matrix(Target target) {
			return classes_vector_to_classes_matrix(target).cast<double>().eval();
		}
	};
}
}
}
#endif