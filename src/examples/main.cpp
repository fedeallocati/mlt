#define EIGEN_USE_MKL_ALL

#include <iostream>

#include <Eigen/Core>

#include "../mlt/models/classifiers/optimizable_linear_classifier.hpp"
#include "../mlt/utils/loss_functions.hpp"

#include "misc.hpp"
#include "../mlt/utils/optimizers/stochastic_gradient_descent.hpp"
#include "../mlt/utils/eigen.hpp"

//#include "../mlt/models/pipeline.hpp"
//#include "../mlt/utils/linear_algebra.hpp"

int main() {
	print_info();
	std::cout << std::endl;

	size_t n_features = 4;
	size_t n_classes = 4;
	size_t n_samples = 1000;
	size_t n_training = 0.8 * n_samples;
	size_t n_val = n_samples - n_training;

	auto input = Eigen::MatrixXd{ Eigen::MatrixXd::Random(n_features, n_samples).cwiseAbs() * 10 };
	auto classes = Eigen::VectorXi(n_samples);

	std::random_device rd;
	std::default_random_engine rng(rd());
	std::uniform_int_distribution<size_t> distribution(0, n_classes);
	//classes = classes.unaryExpr([&] (int i) { return distribution(rng); });<
	for (size_t i = 0; i < n_samples; i++) {
		double sum = input.col(i).sum();
		classes(i) = sum / 10;
	}

	using loss_t = mlt::utils::loss_functions::SoftmaxLoss;
	using opt_t = mlt::utils::optimizers::StochasticGradientDescent<>;

	auto loss = loss_t();
	auto opt = opt_t(200, 500, 0.001, 0.99);

	mlt::models::classifiers::OptimizableLinearClassifier<loss_t, opt_t> sgd_classifier(loss, opt, 5, true);

	sgd_classifier.fit(input.leftCols(n_training), classes.topRows(n_training), false);

	std::cout << "Accuracy: " << sgd_classifier.score(input.rightCols(n_val), classes.bottomRows(n_val)) << std::endl;

	auto classes_matrix = Eigen::MatrixXi{ Eigen::MatrixXi::Zero(classes.maxCoeff() + 1, classes.size()) };

	for (unsigned int i = 0; i < classes.size(); i++) {
		classes_matrix(classes(i), i) = 1;
	}

	auto a = mlt::utils::eigen::tied_random_cols_subset(input, classes_matrix, 10);

	eval_numerical_gradient(sgd_classifier, Eigen::MatrixXd::Random(n_classes, n_features + 1), input, classes_matrix.cast<double>());

	std::cin.get();

	return 0;
}