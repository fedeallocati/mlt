#ifndef MLT_MODELS_REGRESSORS_LEAST_SQUARES_LINEAR_REGRESSION_HPP
#define MLT_MODELS_REGRESSORS_LEAST_SQUARES_LINEAR_REGRESSION_HPP

#include <Eigen/Core>

#include "../linear_model.hpp"
#include "../../utils/linalg.hpp"

namespace mlt {
namespace models {
namespace regressors {
    class LeastSquaresLinearRegression : public LinearModel {
    public:         
        explicit LeastSquaresLinearRegression(bool fit_intercept) : LinearModel(fit_intercept) {}

        LeastSquaresLinearRegression& fit(const Eigen::MatrixXd& input, const Eigen::MatrixXd& target) {
            // Closed-form solution of Least Squares Linear Regression: pinv(input' * input) * input' * target            
			Eigen::MatrixXd input_prime(input.rows(), input.cols() + (_fit_intercept ? 1 : 0));
			input_prime.leftCols(input.cols()) << input;

			if (_fit_intercept) {
				input_prime.rightCols<1>() = Eigen::VectorXd::Ones(input.rows());
			}

            Eigen::MatrixXd coeffs = utils::linalg::pseudo_inverse(input_prime.transpose() * input_prime) * input_prime.transpose() * target;

			if (_fit_intercept) {
				_set_coefficients_and_intercepts(coeffs.topRows(coeffs.rows() - 1), coeffs.bottomRows<1>());
			}
			else {
				_set_coefficients(coeffs);
			}

			return *this;
        }
    };
}
}
}
#endif