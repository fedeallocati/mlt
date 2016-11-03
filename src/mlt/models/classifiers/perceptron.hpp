#ifndef MLT_MODELS_CLASSIFIERS_PERCEPTRON_HPP
#define MLT_MODELS_CLASSIFIERS_PERCEPTRON_HPP

#include <algorithm>
#include <Eigen/Core>

#include "linear_classifier.hpp"
#include "../../utils/linear_algebra.hpp"

namespace mlt {
namespace models {
namespace classifiers {
template <class Rng>
class Perceptron : public LinearClassifier<Perceptron<Rng>> {
public:
	template <class R, class = enable_if<is_same<decay_t<R>, Rng>::value>>
	Perceptron(size_t epochs,  bool shuffle, double learning_rate, bool fit_intercept, R&& rng) : LinearClassifier(fit_intercept),
		_epochs(epochs), _shuffle(shuffle), _learning_rate(learning_rate), _rng(forward<R>(rng)) {}

	Self& fit(Features input, Target classes, bool cold_start = true) {
		auto n_classes = classes.maxCoeff() + 1;
		auto n_features = input.rows();
		auto n_samples = input.cols();

		MatrixXd current_coeffs = _fitted && !cold_start && num_classes() == n_classes ?
		                                coefficients() :
		                                (MatrixXd::Zero(n_classes, n_features + (fit_intercept() ? 1 : 0)) * 0.005);

		MatrixXd input_prime(input.rows() + (fit_intercept() ? 1 : 0), n_samples);
		input_prime.topRows(input.rows()) << input;

		if (fit_intercept()) {
			input_prime.bottomRows<1>() = VectorXd::Ones(n_samples);
		}

		vector<int> idxs;
		if (_shuffle) {
			idxs = vector<int>(n_samples);
			for (auto i = 0; i < n_samples; i++) {
				idxs[i] = i;
			}
		}

		for (auto epoch = 0; epoch < _epochs; epoch++) {
			if (_shuffle) {
				shuffle(idxs.begin(), idxs.end(), _rng);
			}

			for (auto i = 0; i < n_samples; i++) {
				auto idx = _shuffle ? idxs[i] : i;

				auto f = input.col(idx).eval();
				auto t = classes(idx);
				auto y = max_row(_apply_linear_transformation(f, current_coeffs));

				if (t != y) {
					auto f_t = f.transpose().eval();
					current_coeffs.block(y, 0, 1, n_features) -= _learning_rate * f_t;
					current_coeffs.block(t, 0, 1, n_features) += _learning_rate * f_t;

					if (fit_intercept()) {
						current_coeffs(y, n_classes) -= 1;
						current_coeffs(t, n_classes) += 1;
					}					
				}
			}
		}

		_set_coefficients(current_coeffs);

		return *this;
	}

protected:
	size_t _epochs;
	bool _shuffle;
	double _learning_rate;
	Rng& _rng;
};

	template <class R = default_random_engine>
	auto create_perceptron(size_t epoch = 100, bool shuffle = true, double learning_rate = 0.001, bool fit_intercept = true, R&& rng = default_random_engine()) {
		return Perceptron<R>(epoch, shuffle, learning_rate, fit_intercept, forward<R>(rng));
	}
}
}
}
#endif