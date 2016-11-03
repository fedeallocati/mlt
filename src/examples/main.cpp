#define EIGEN_USE_MKL_ALL

#include <iostream>
#include <iterator>
#include <random>

#include <Eigen/Core>

#include "models/classifiers/optimizable_linear_classifier.hpp"
#include "utils/optimizers/stochastic_gradient_descent.hpp"
#include "utils/loss_functions.hpp"
#include "utils/eigen.hpp"
#include "../misc.hpp"

using namespace std;
using namespace Eigen;

using namespace mlt::models::classifiers;
using namespace mlt::utils::optimizers;
using namespace mlt::utils::loss_functions;
using namespace mlt::utils::eigen;

int main() {
	print_info();
	cout << endl;

	auto n_features = 5;
	auto n_classes = 4;
	auto n_samples = 1000;
	auto n_training = 0.8 * n_samples;
	auto n_val = n_samples - n_training;

	auto input = MatrixXd{ MatrixXd::Random(n_features, n_samples).cwiseAbs() * 10 };
	auto classes = VectorXi(n_samples);

	for (auto i = 0; i < n_samples; i++) {
		auto sum = input.col(i).sum();
		classes(i) = sum / 10;
	}

	using loss_t = SoftmaxLoss;
	using opt_t = StochasticGradientDescent<>;

	auto loss = loss_t();
	auto opt = opt_t(200, 500, 0.001, 0.99);

	OptimizableLinearClassifier<loss_t, opt_t> model(loss, opt, 5, true);

	model.fit(input.leftCols(n_training), classes.topRows(n_training), false);

	cout << "Accuracy: " << model.score(input.rightCols(n_val), classes.bottomRows(n_val)) << endl;

	eval_numerical_gradient(model, MatrixXd::Random(n_classes, n_features + 1), input, classes_vector_to_classes_matrix(classes).cast<double>());

	cin.get();

	return 0;
}