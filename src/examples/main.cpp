#define EIGEN_USE_MKL_ALL

#include <iostream>

#include <Eigen/Core>

#include "../mlt/models/transformers/autoencoder.hpp"
#include "../mlt/utils/optimizers/stochastic_gradient_descent.hpp"
#include "../mlt/utils/activation_functions.hpp"

#include "misc.hpp"

int main() {
	print_info();
	std::cout << std::endl;

	auto features = 10;
	auto samples = 1000;
	auto hidden = 4;
	Eigen::MatrixXd input = (Eigen::MatrixXd::Random(features, samples).array() + 1) / 2;
	Eigen::MatrixXd coeffs = Eigen::MatrixXd::Random(hidden + 1, features + 1) * 4 / std::sqrt(6.0 / (hidden + features));
	
	mlt::utils::optimizers::StochasticGradientDescent<> sgd(1, 2);
	mlt::utils::activation_functions::SigmoidActivation activation;
	mlt::models::transformers::Autoencoder<mlt::utils::activation_functions::SigmoidActivation, mlt::utils::optimizers::StochasticGradientDescent<>> autoencoder(hidden, activation, sgd, 0);

	eval_numerical_gradient(autoencoder, coeffs, input, input);
	std::cin.get();

	return 0;
}
