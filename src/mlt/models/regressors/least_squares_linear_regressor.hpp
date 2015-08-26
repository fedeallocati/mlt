#ifndef LEAST_SQUARES_LINEAR_REGRESSOR_HPP
#define LEAST_SQUARES_LINEAR_REGRESSOR_HPP

#include <Eigen/Core>
#include <Eigen/SVD>

namespace mlt {
namespace models {
namespace regressors {
    
    // Implementation of Least Squares Linear Regression
    // Categorization: 
    // - Application: Regressor
    // - Parametrization: Parametrized
    // - Method of Training: Self-Trainable, Derivative-Free, Gradient-Based   
    class LeastSquaresLinearRegressor {
	public:			
		LeastSquaresLinearRegressor() : _input(0) {}

		LeastSquaresLinearRegressor(size_t input) : _input(input), _beta(Eigen::VectorXd::Zero(input + 1)) {}

		// Disable copy constructors
		LeastSquaresLinearRegressor(const LeastSquaresLinearRegressor& other) = delete;
		LeastSquaresLinearRegressor& operator=(const LeastSquaresLinearRegressor& other) = delete;

		inline size_t input() const {
			assert(_input != 0);
			return _input;
		}

		inline size_t output() const {
			return 1;
		}

		inline bool add_intercept() const {
			return true;
		}

		inline bool is_initialized() const {
			return _input != 0;
		}

		inline void reset() {
			assert(_input != 0);
			_beta.setZero();
		}

		inline Eigen::VectorXd regress_single(const Eigen::VectorXd& input) const {
			return _beta.transpose() * input;
        }

		inline Eigen::VectorXd regress_multi(const Eigen::MatrixXd& input) const {
			return input * _beta;
		}

		inline size_t params_size() const {
			assert(_input != 0);
			return _input + 1;
		}

		inline const Eigen::VectorXd& params() const {
			return _beta;
		}

		inline void set_params(const Eigen::VectorXd& beta) {
			_beta = beta;
			_input = beta.size() - 1;
		}

		inline double cost(const Eigen::MatrixXd& input, const Eigen::MatrixXd& result) const {
			return cost(_beta, input, result);
		}

		inline double cost(const Eigen::VectorXd& beta, const Eigen::MatrixXd& input, const Eigen::MatrixXd& result) const {
			return ((input * beta) - result).squaredNorm() / (2 * result.rows());
		}
		
		inline Eigen::VectorXd cost_gradient(const Eigen::MatrixXd& input, const Eigen::MatrixXd& result) const {
			return cost_gradient(_beta, input, result);
		}

		inline Eigen::VectorXd cost_gradient(const Eigen::VectorXd& beta, const Eigen::MatrixXd& input, const Eigen::MatrixXd& result) const {
			return (input.transpose() * ((input * beta) - result)) / result.rows();
		}

		void self_train(const Eigen::MatrixXd& input, const Eigen::MatrixXd& result, bool reset = false) {
			// Moore-Penrose pseudoinverse
			double epsilon = 1e-9;
			Eigen::JacobiSVD<Eigen::MatrixXd> svd(input.transpose() * input, Eigen::ComputeThinU | Eigen::ComputeThinV);
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
			_input = _beta.size() - 1;
		}	

    protected:		
		size_t _input;
		Eigen::VectorXd _beta;
	};
}
}
}
#endif