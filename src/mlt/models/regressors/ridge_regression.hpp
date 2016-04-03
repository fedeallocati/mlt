#ifndef MLT_MODELS_REGRESSORS_RIDGE_REGRESSION_HPP
#define MLT_MODELS_REGRESSORS_RIDGE_REGRESSION_HPP

#include <Eigen/Core>

#include "../linear_model.hpp"
#include "../../utils/linalg.hpp"

namespace mlt {
namespace models {
namespace regressors {
	class RidgeRegression : public LinearModel {
	public:
		explicit RidgeRegression(double regularization, bool fit_intercept) : _regularization(regularization), LinearModel(fit_intercept) {}

		RidgeRegression& fit(const Eigen::MatrixXd& input, const Eigen::MatrixXd& target) {
			// Closed-form solution of Ridge Linear Regression: pinv((input' * input) + I * regularization) * input' * target            
			Eigen::MatrixXd input_prime(input.rows(), input.cols() + (_fit_intercept ? 1 : 0));
			input_prime.leftCols(input.cols()) << input;

			if (_fit_intercept) {
				input_prime.rightCols<1>() = Eigen::VectorXd::Ones(input.rows());
			}

			Eigen::MatrixXd reg = Eigen::MatrixXd::Identity(input_prime.cols(), input_prime.cols()) * _regularization;
			reg(reg.rows() -1, reg.cols() - 1) = 0;

			Eigen::MatrixXd coeffs = utils::linalg::pseudo_inverse(
				(input_prime.transpose() * input_prime) + reg)
				* input_prime.transpose() * target;

			if (_fit_intercept) {
				_set_coefficients_and_intercepts(coeffs.topRows(coeffs.rows() - 1), coeffs.bottomRows<1>());
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
	};
}
}
}
#endif