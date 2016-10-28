#define EIGEN_USE_MKL_ALL

#include <iostream>
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

	size_t n_features = 4;
	size_t n_classes = 4;
	size_t n_samples = 1000;
	size_t n_training = 0.8 * n_samples;
	size_t n_val = n_samples - n_training;

	auto input = MatrixXd{ MatrixXd::Random(n_features, n_samples).cwiseAbs() * 10 };
	auto classes = VectorXi(n_samples);

	random_device rd;
	default_random_engine rng(rd());
	uniform_int_distribution<size_t> distribution(0, n_classes);
	//classes = classes.unaryExpr([&] (int i) { return distribution(rng); });<
	for (size_t i = 0; i < n_samples; i++) {
		double sum = input.col(i).sum();
		classes(i) = sum / 10;
	}

	using loss_t = SoftmaxLoss;
	using opt_t = StochasticGradientDescent<>;

	auto loss = loss_t();
	auto opt = opt_t(200, 500, 0.001, 0.99);

	OptimizableLinearClassifier<loss_t, opt_t> sgd_classifier(loss, opt, 5, true);

	sgd_classifier.fit(input.leftCols(n_training), classes.topRows(n_training), false);

	cout << "Accuracy: " << sgd_classifier.score(input.rightCols(n_val), classes.bottomRows(n_val)) << endl;

	auto classes_matrix = MatrixXi{ MatrixXi::Zero(classes.maxCoeff() + 1, classes.size()) };

	for (unsigned int i = 0; i < classes.size(); i++) {
		classes_matrix(classes(i), i) = 1;
	}

	eval_numerical_gradient(sgd_classifier, MatrixXd::Random(n_classes, n_features + 1), input, classes_matrix.cast<double>());

	cin.get();

	return 0;
}