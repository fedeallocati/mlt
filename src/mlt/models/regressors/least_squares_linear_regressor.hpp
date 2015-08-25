#ifndef LEAST_SQUARES_LINEAR_REGRESSOR_HPP
#define LEAST_SQUARES_LINEAR_REGRESSOR_HPP

#include <Eigen/Core>
#include <Eigen/SVD>

namespace mlt {
namespace models {
namespace regressors {
    
    // Implementation of Least Squares Linear Regression
    // Categories: 
    // - Application: Regressor
    // - Parametrization: Parametrized
    // - Method of Training: Self-Trainable, Derivative-Free, Gradient-Based   
    template <typename Params>
    class LeastSquaresLinearRegressor {
    public:        
		typedef Params::LeastSquaresLinearRegression params_t;
		typedef Eigen::Matrix<double, params_t::size + 1, 1> beta_t;
		typedef Eigen::Matrix<double, params_t::size + 1, 1> single_input_t;
		typedef Eigen::Matrix<double, 1, 1> single_result_t;
		typedef Eigen::Matrix<double, Eigen::Dynamic, params_t::size + 1> multiple_input_t;
		typedef Eigen::Matrix<double, Eigen::Dynamic, 1> multiple_result_t;

		const static size_t input = params_t::size;
		const static bool add_intercept = true;
		const static size_t output = 1;
		const static size_t params_size = params_t::size + 1;

		inline void init() {
			_beta = beta_t::Zero();
		}

		inline single_result_t regress(single_input_t input) const {			
			return _beta.transpose() * input;
        }

		inline multiple_result_t regress(multiple_input_t input) const {
			return input * _beta;
		}

		inline const beta_t& params() const {
			return _beta;
		}

		inline void set_params(const beta_t& beta) {
			_beta = beta;
		}

		inline double cost(multiple_input_t input, multiple_result_t result) const {
			return cost(_beta, input, result);
		}

		inline double cost(beta_t beta, multiple_input_t input, multiple_result_t result) const {
			return ((input * beta) - result).array().pow(2).sum() / (2 * result.rows());
		}

		inline beta_t cost_gradient(multiple_input_t input, multiple_result_t result) const {
			return cost_gradient(_beta, input, result);
		}

		inline beta_t cost_gradient(beta_t beta, multiple_input_t input, multiple_result_t result) const {
			return (1 / result.rows()) * (input.transpose() * ((input * beta) - result));
		}

		void self_train(multiple_input_t input, multiple_result_t result, bool reset = false) {
			// Moore-Penrose pseudoinverse
			double epsilon = 1e-9;
			Eigen::JacobiSVD<MatrixXd> svd(input.transpose() * input, Eigen::ComputeThinU | Eigen::ComputeThinV);
			const auto singVals = svd.singularValues();
			auto invSingVals = singVals;
			for (int i = 0; i < singVals.rows(); i++) {
				if (singVals(i) <= epsilon) {
					invSingVals(i) = 0.0;
				} else {
					invSingVals(i) = 1.0 / invSingVals(i);
				}
			}
			Eigen::MatrixXd inv = (svd.matrixV() * invSingVals.asDiagonal() * svd.matrixU().transpose());

			// Closed-form solution of Least Squares Linear Regression
			_beta = inv * input.transpose() * result;
		}	

    protected:
		beta_t _beta;
	};
}
}
}
#endif