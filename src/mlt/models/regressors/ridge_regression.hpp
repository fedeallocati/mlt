#ifndef MLT_MODELS_REGRESSORS_RIDGE_REGRESSION_HPP
#define MLT_MODELS_REGRESSORS_RIDGE_REGRESSION_HPP

#include <Eigen/Core>

#include "linear_regressor_model.hpp"
#include "../../utils/linear_solvers.hpp"

namespace mlt {
namespace models {
namespace regressors {
	template <class Solver = utils::linear_solvers::SVDSolver>
	class RidgeRegression : public LinearRegressorModel {
	public:
		explicit RidgeRegression(double regularization, bool fit_intercept = true) : LinearRegressorModel(fit_intercept),
			_regularization(regularization), _solver(Solver()) {}

		explicit RidgeRegression(double regularization, const Solver& solver, bool fit_intercept = true) : LinearRegressorModel(fit_intercept),
			_regularization(regularization), _solver(solver) {}

		explicit RidgeRegression(double regularization, Solver&& solver, bool fit_intercept = true) : LinearRegressorModel(fit_intercept),
			_regularization(regularization), _solver(solver) {}

		RidgeRegression& fit(const Eigen::MatrixXd& input, const Eigen::MatrixXd& target) {
			Eigen::MatrixXd input_prime(input.rows() + (_fit_intercept ? 1 : 0), input.cols());
			input_prime.topRows(input.rows()) << input;
			Eigen::MatrixXd reg = Eigen::MatrixXd::Identity(input_prime.rows(), input_prime.rows()) * _regularization;

			if (_fit_intercept) {
				input_prime.bottomRows<1>() = Eigen::VectorXd::Ones(input.cols());
				reg(reg.rows() - 1, reg.cols() - 1) = 0;
			}

			_set_coefficients(_solver.compute((input_prime * input_prime.transpose() + reg)).solve(input_prime * target.transpose()).transpose());

			return *this;
		}

	protected:
		double _regularization;
		Solver _solver;
	};
}
}
}
#endif