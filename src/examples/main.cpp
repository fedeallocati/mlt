#define EIGEN_USE_MKL_ALL

#include <iostream>

#include <Eigen/Core>

/*#include "../mlt/models/transformers/autoencoder.hpp"
#include "../mlt/utils/optimizers/stochastic_gradient_descent.hpp"
#include "../mlt/utils/activation_functions.hpp"
#include "../mlt/models/pipeline.hpp"
#include "../mlt/models/classifiers/optimizable_linear_classifier.hpp"
#include "../mlt/utils/loss_functions.hpp"*/

#include "misc.hpp"


template <typename Concrete, typename Param>
class A
{
protected:
	template <typename P, class = std::enable_if<std::is_same<std::decay_t<P>, Param>::value>>
	explicit A(P&& param, bool fit_intercept) :
		_p(std::forward<P>(param)), _fit_intercept(fit_intercept) {}

	Param _p;
	bool _fit_intercept;
};

template <typename Param>
class B : public A<B<Param>, Param>
{
public:
	using A<B<Param>, Param>::A<B<Param>, Param>;
};

int main() {
	print_info();
	std::cout << std::endl;

	B<int> b(1, true);

	/*auto features = 10;
	auto samples = 1000;
	auto hidden = 4;
	Eigen::MatrixXd input = (Eigen::MatrixXd::Random(features, samples).array() + 1) / 2;
	Eigen::MatrixXd coeffs = Eigen::MatrixXd::Random(hidden + 1, features + 1) * 4 / std::sqrt(6.0 / (hidden + features));
	
	mlt::utils::optimizers::StochasticGradientDescent<> sgd(1, 2);
	mlt::utils::activation_functions::SigmoidActivation activation;
	mlt::models::transformers::Autoencoder<mlt::utils::activation_functions::SigmoidActivation, mlt::utils::optimizers::StochasticGradientDescent<>> autoencoder(hidden, activation, sgd, 0);

	eval_numerical_gradient(autoencoder, coeffs, input, input);
	std::cin.get();

	auto pipeline = mlt::models::create_pipeline(autoencoder, autoencoder, autoencoder);
	pipeline.fit(input, true);

	mlt::utils::optimizers::StochasticGradientDescent<> grad_descent(10, 2000, 0.001, 1);
	mlt::utils::loss_functions::SquaredLoss loss;
	mlt::models::classifiers::OptimizableLinearClassifier<mlt::utils::loss_functions::SquaredLoss, mlt::utils::optimizers::StochasticGradientDescent<>> sgd_classifier(loss, grad_descent, 0.5, true);
	Eigen::MatrixXi output = Eigen::MatrixXi::Random(3, samples);
	sgd_classifier.fit(input, output, true);

	mlt::models::classifiers::OptimizableLinearClassifier<mlt::utils::loss_functions::SquaredLoss, mlt::utils::optimizers::StochasticGradientDescent<>> l2(loss, sgd, 1.0);*/

	return 0;
}
