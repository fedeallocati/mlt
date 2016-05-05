#define EIGEN_USE_MKL_ALL

#include <iostream>

#include <Eigen/Core>

#include "../mlt/models/transformers/autoencoder.hpp"
#include "../mlt/models/transformers/sparse_autoencoder.hpp"
#include "../mlt/models/transformers/tied_autoencoder.hpp"
#include "../mlt/models/transformers/sparse_tied_autoencoder.hpp"
#include "../mlt/utils/optimizers/stochastic_gradient_descent.hpp"
#include "../mlt/utils/activation_functions.hpp"

#include "misc.hpp"

void autoencoder_examples() {
	print_info();
	std::cout << std::endl;

	auto features = 10;
	auto samples = 2000;
	auto hidden = 4;
	auto output = 10;

	Eigen::MatrixXd input = (Eigen::MatrixXd::Random(features, samples).array() + 1) / 2;
	Eigen::MatrixXd coeffs = Eigen::MatrixXd::Random(output, features) * 4 / std::sqrt(6.0 / (hidden + features));
	Eigen::VectorXd encoderCoeffs = Eigen::VectorXd::Random(hidden * features + hidden + features * hidden + features) * 4 / std::sqrt(6.0 / (hidden + features));
	Eigen::VectorXd tiedEncoderCoeffs = Eigen::VectorXd::Random(hidden * features + hidden + features) * 4 / std::sqrt(6.0 / (hidden + features));

	using activation_t = mlt::utils::activation_functions::SigmoidActivation;
	using optimizer_t = mlt::utils::optimizers::StochasticGradientDescent<>;

	optimizer_t sgd(1, 20);
	activation_t activation;

	auto autoencoder1 = mlt::models::transformers::create_autoencoder(hidden, activation, activation, sgd, 3e-3);
	eval_numerical_gradient(autoencoder1, encoderCoeffs, input, input);
	std::cin.get();

	auto autoencoder2 = mlt::models::transformers::create_sparse_autoencoder(hidden, activation, activation, sgd, 3e-3, 0.1, 3);
	eval_numerical_gradient(autoencoder2, encoderCoeffs, input, input);
	std::cin.get();

	auto autoencoder3 = mlt::models::transformers::create_tied_autoencoder(hidden, activation, activation, sgd, 3e-3);
	eval_numerical_gradient(autoencoder3, tiedEncoderCoeffs, input, input);
	std::cin.get();

	auto autoencoder4 = mlt::models::transformers::create_sparse_tied_autoencoder(hidden, activation, activation, sgd, 3e-3, 0.1, 3);
	eval_numerical_gradient(autoencoder4, tiedEncoderCoeffs, input, input);
	std::cin.get();

	autoencoder1.fit(input);
	autoencoder1.fit(input, false);

	std::cin.get();

	autoencoder2.fit(input);
	autoencoder2.fit(input, false);

	std::cin.get();

	autoencoder3.fit(input);
	autoencoder3.fit(input, false);

	std::cin.get();

	autoencoder4.fit(input);
	autoencoder4.fit(input, false);

	std::cin.get();
}