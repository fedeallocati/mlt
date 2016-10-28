#ifndef MLT_MODELS_REGRESSORS_RIDGE_REGRESSION_HPP
#define MLT_MODELS_REGRESSORS_RIDGE_REGRESSION_HPP

#include <type_traits>

#include <Eigen/Core>

#include "linear_regressor.hpp"
#include "../../utils/linear_solvers.hpp"

namespace mlt {
namespace models {
namespace regressors {
	using namespace utils::linear_solvers;

	template <class Solver = SVDSolver>
	class RidgeRegression : public LinearRegressor<RidgeRegression<Solver>> {
	public:
		explicit RidgeRegression(double regularization, bool fit_intercept = true) : LinearRegressor(fit_intercept),
			_regularization(regularization), _solver(Solver()) {}

		template <class S, class = enable_if<is_same<decay_t<S>, Solver>::value>>
		explicit RidgeRegression(double regularization, S&& solver, bool fit_intercept = true) : LinearRegressor(fit_intercept),
			_regularization(regularization), _solver(forward<S>(solver)) {}

		Self& fit(Features input, Target target, bool = true) {
			MatrixXd input_prime(input.rows() + (_fit_intercept ? 1 : 0), input.cols());
			input_prime.topRows(input.rows()) << input;
			MatrixXd reg = MatrixXd::Identity(input_prime.rows(), input_prime.rows()) * _regularization;

			if (_fit_intercept) {
				input_prime.bottomRows<1>() = VectorXd::Ones(input.cols());
				reg(reg.rows() - 1, reg.cols() - 1) = 0;
			}

			_set_coefficients(_solver.compute((input_prime * input_prime.transpose() + reg)).solve(input_prime * target.transpose()).transpose());

			return _self();
		}

	protected:
		double _regularization;
		Solver _solver;
	};
}
}
}
#endif