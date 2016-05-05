#define EIGEN_USE_MKL_ALL

#include <iostream>

#include <Eigen/Core>

#include "misc.hpp"

#include "../mlt/models/regressors/least_squares_linear_regression.hpp"
#include "../mlt/models/regressors/ridge_regression.hpp"
#include "../mlt/models/regressors/optimizable_linear_regressor.hpp"
#include "../mlt/utils/optimizers/stochastic_gradient_descent.hpp"
#include "../mlt/utils/loss_functions.hpp"

void benchmark_linear_solvers() {
	Eigen::MatrixXd XN = Eigen::MatrixXd::Random(100, 100000);
	Eigen::RowVectorXd YN = Eigen::RowVectorXd::Random(100000);

	mlt::models::regressors::LeastSquaresLinearRegression<mlt::utils::linear_solvers::SVDSolver> linear_regressor_svd(false);
	benchmark(linear_regressor_svd, XN, YN, 100);
	mlt::models::regressors::LeastSquaresLinearRegression<mlt::utils::linear_solvers::LDLTSolver> linear_regressor_ldlt(false);
	benchmark(linear_regressor_ldlt, XN, YN, 100);
	std::cout << std::endl;
	mlt::models::regressors::LeastSquaresLinearRegression<mlt::utils::linear_solvers::CGSolver> linear_regressor_cg(false);
	benchmark(linear_regressor_cg, XN, YN, 100);
	std::cout << std::endl;

	std::cout << "Diff: " << (linear_regressor_svd.coefficients() - linear_regressor_ldlt.coefficients()).squaredNorm() << std::endl;
	std::cout << "Diff: " << (linear_regressor_svd.coefficients() - linear_regressor_cg.coefficients()).squaredNorm() << std::endl;
	std::cout << "Diff: " << (linear_regressor_ldlt.coefficients() - linear_regressor_cg.coefficients()).squaredNorm() << std::endl;
}

void test_optimizable_linear_regressors() {
	auto samples = 100;
	Eigen::MatrixXd input = Eigen::MatrixXd::Random(3, samples) * 100;
	Eigen::MatrixXd output = Eigen::MatrixXd::Random(2, samples).array();
	output = (output.array() > 0.0).cast<double>();
	output.row(1) = 1 - output.row(0).array();

	mlt::utils::optimizers::StochasticGradientDescent<> sgd;
	mlt::utils::loss_functions::SquaredLoss loss;

	mlt::models::regressors::OptimizableLinearRegressor<mlt::utils::loss_functions::SquaredLoss, mlt::utils::optimizers::StochasticGradientDescent<>> model(loss, sgd, 0, false);
	eval_numerical_gradient(model, Eigen::MatrixXd::Random(2, 3) * 0.05, input, output);

	mlt::models::regressors::OptimizableLinearRegressor<mlt::utils::loss_functions::SquaredLoss, mlt::utils::optimizers::StochasticGradientDescent<>> model2(loss, sgd, 0, true);
	eval_numerical_gradient(model2, Eigen::MatrixXd::Random(2, 4) * 0.05, input, output);
}

void lr_examples() {
	benchmark_linear_solvers();

	Eigen::MatrixXd X1(2, 3);
	Eigen::MatrixXd Y1(1, 3);

	X1.row(0) << 0, 1, 2;
	X1.row(1) << 0, 1, 2;
	Y1 << 0, 1, 2;

	mlt::models::regressors::LeastSquaresLinearRegression<> linear_regressor(false);
	linear_regressor.fit(X1, Y1);

	std::cout << "LinearRegression: " << std::endl;
	std::cout << linear_regressor.coefficients() << std::endl;
	if (linear_regressor.fit_intercept()) {
	std::cout << linear_regressor.intercepts() << std::endl;
	}
	std::cout << linear_regressor.predict(X1.col(0)) << std::endl;

	Eigen::MatrixXd X2(2, 3);
	Eigen::MatrixXd Y2(1, 3);

	X2.row(0) << 0, 0, 1;
	X2.row(1) << 0, 0, 1;
	Y2 << 0, .1, 1;

	mlt::models::regressors::RidgeRegression<> ridge_regressor(0.5, true);
	ridge_regressor.fit(X2, Y2);

	std::cout << "RidgeRegression: " << std::endl;
	std::cout << ridge_regressor.coefficients() << std::endl;
	std::cout << ridge_regressor.intercepts() << std::endl;
	std::cout << ridge_regressor.predict(X2.col(0)) << std::endl;

	mlt::utils::optimizers::StochasticGradientDescent<> grad_descent(10, 2000, 0.001, 1);
	mlt::utils::loss_functions::SquaredLoss loss;
	mlt::models::regressors::OptimizableLinearRegressor<mlt::utils::loss_functions::SquaredLoss, mlt::utils::optimizers::StochasticGradientDescent<>> sgd(loss, grad_descent, 0.5, true);

	sgd.fit(X2, Y2, true);

	std::cout << "OptimizableLinearRegressor<SquaredLoss, SGD>: " << std::endl;
	std::cout << ridge_regressor.coefficients() << std::endl;
	std::cout << ridge_regressor.intercepts() << std::endl;
	std::cout << sgd.predict(X2.col(0)) << std::endl;

	std::cout << "loss with closed form: " << sgd.loss(ridge_regressor.all_coefficients(), X2, Y2) << std::endl;
	std::cout << "loss with SGD: " << sgd.loss(sgd.all_coefficients(), X2, Y2) << std::endl;
}