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
			Eigen::MatrixXd input_prime(input.rows() + (fit_intercept() ? 1 : 0), input.cols());
			input_prime.topRows(input.rows()) << input;

			if (fit_intercept()) {
				input_prime.bottomRows<1>() = Eigen::VectorXd::Ones(input.cols());
			}

			Eigen::MatrixXd reg = Eigen::MatrixXd::Identity(input_prime.rows(), input_prime.rows()) * _regularization;
			reg(reg.rows() -1, reg.cols() - 1) = 0;

			Eigen::MatrixXd coeffs = _solver.compute((input_prime * input_prime.transpose() + reg)).solve(input_prime * target.transpose()).transpose();

			if (fit_intercept()) {
				_set_coefficients_and_intercepts(coeffs.leftCols(coeffs.cols() - 1), coeffs.rightCols<1>());
			}
			else {
				_set_coefficients(coeffs);
			}

			return *this;
		}

	protected:
		// TODO: Implement different training methods
		inline double _cost_internal(const Eigen::MatrixXd& beta, const Eigen::MatrixXd& input, const Eigen::MatrixXd& result) const {
			double loss = ((input * beta) - result).array().pow(2).sum() / (2 * input.rows());
			loss += _regularization * (beta.array().pow(2)).sum();

			return loss;
		}

		inline std::tuple<double, Eigen::MatrixXd> _cost_and_gradient_internal(const Eigen::MatrixXd& beta, const Eigen::MatrixXd& input, const Eigen::MatrixXd& result) const {
			Eigen::MatrixXd diff = input * beta - result;
			double loss = diff.array().pow(2).sum() / (2 * input.rows());
			loss += _regularization * (beta.array().pow(2)).sum();

			Eigen::MatrixXd d_beta = (input.transpose() * diff) / input.rows();
			d_beta += _regularization * 2 * beta;

			return std::make_tuple(loss, d_beta);
		}

		double _regularization;
		Solver _solver;
	};
}
}
}
#endif