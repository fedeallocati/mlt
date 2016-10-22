#ifndef MLT_MODELS_PERCEPTRON_CLASSIFIER_HPP
#define MLT_MODELS_PERCEPTRON_CLASSIFIER_HPP

#include <Eigen/Core>

#include "linear_classifier.hpp"

namespace mlt {
namespace models {
namespace classifiers {
	class PerceptronClassifier : public LinearClassifier<PerceptronClassifier> {
	public:
		explicit PerceptronClassifier(size_t epochs = 1000, bool fit_intercept = true) : LinearClassifier(fit_intercept), _epochs(epochs) {}

		PerceptronClassifier& fit(const Eigen::Ref<const Eigen::MatrixXd>& input, const Eigen::Ref<const Eigen::VectorXi>& classes, bool cold_start = true) {
			Eigen::MatrixXd current_coefficients = this->_fitted && !cold_start ?
				this->coefficients() :
				(Eigen::MatrixXd::Random(target.rows(), input.rows() + (this->fit_intercept() ? 1 : 0)) * 0.005);

			Eigen::MatrixXd input_prime(input.rows() + (_fit_intercept ? 1 : 0), input.cols());
			input_prime.topRows(input.rows()) << input;

			if (_fit_intercept) {
				input_prime.bottomRows<1>() = Eigen::VectorXd::Ones(input.cols());
			}

			auto samples = input.cols();

			this->_set_coefficients(current_coefficients);
			for (auto epoch = 0; epoch < _epochs; epoch++) {
				for (auto i = 0; i < samples; i++) {
					auto activation = _apply_linear_transformation(input.col(i))();
					if (target(i) == 0 && activation >= 0) {

					} else if (target(i) == 0 && activation >= 0)
				}
			}
			return *this;
		}
	protected:
		size_t _epochs;
	};
}
}
}
#endif