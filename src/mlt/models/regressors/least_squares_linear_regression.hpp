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
        explicit LeastSquaresLinearRegression(bool fit_intercept = true) : LinearModel(fit_intercept) {}

        LeastSquaresLinearRegression& fit(const Eigen::MatrixXd& input, const Eigen::MatrixXd& target) {
            // Closed-form solution of Least Squares Linear Regression: pinv(input' * input) * input' * target            
			Eigen::MatrixXd input_prime(input.rows() + (_fit_intercept ? 1 : 0), input.cols());
			input_prime.topRows(input.rows()) << input;

			if (_fit_intercept) {
				input_prime.bottomRows<1>() = Eigen::VectorXd::Ones(input.cols());
			}

			// target = 1 * 3
			// input = 2 * 3
			// inv = 2 * 2
			// inv * input * target' = ((2 * 2) * (2 * 3)) * (3 * 1) = (2 * 3) * (3 * 1) = 2 * 1
			Eigen::MatrixXd coeffs = (utils::linalg::pseudo_inverse(input_prime * input_prime.transpose())
				* input_prime * target.transpose()).transpose();
			
			if (_fit_intercept) {
				_set_coefficients_and_intercepts(coeffs.leftCols(coeffs.cols() - 1), coeffs.rightCols<1>());
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