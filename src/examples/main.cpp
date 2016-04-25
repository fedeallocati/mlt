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

	auto features = 2;
	auto samples = 5;
	auto hidden = 2;
	Eigen::MatrixXd input = (Eigen::MatrixXd::Random(features, samples).array() + 1) / 2;
	
	mlt::utils::optimizers::StochasticGradientDescent<> sgd;
	mlt::utils::activation_functions::SigmoidActivation activation;
	mlt::models::transformers::Autoencoder<mlt::utils::activation_functions::SigmoidActivation, mlt::utils::optimizers::StochasticGradientDescent<>> autoencoder(hidden, activation, sgd, 0);
	//autoencoder.fit(input);
	
	eval_numerical_gradient(autoencoder, Eigen::MatrixXd::Random(hidden + 1, features + 1) * 4 / std::sqrt(6.0 / (hidden + features)), input, input);

	std::cin.get();

	return 0;
}
