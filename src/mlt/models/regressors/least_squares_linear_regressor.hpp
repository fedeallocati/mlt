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
		LeastSquaresLinearRegressor() : _init(false) {}

		LeastSquaresLinearRegressor(size_t input, size_t output) : _init(true), _beta(Eigen::MatrixXd::Zero(input + 1, output)) {}

		// Disable copy constructors
		LeastSquaresLinearRegressor(const LeastSquaresLinearRegressor& other) = delete;
		LeastSquaresLinearRegressor& operator=(const LeastSquaresLinearRegressor& other) = delete;

		inline size_t input() const {
			assert(_init);
			return _beta.rows() - 1;
		}

		inline size_t output() const {
			assert(_init);
			return _beta.cols();
		}

		inline bool add_intercept() const {
			return true;
		}

		inline bool is_initialized() const {
			return _init;
		}

		inline void init(size_t input, size_t output) {
			_beta = Eigen::MatrixXd::Zero(input + 1, output);
			_init = true;
		}

		inline void reset() {
			assert(_init);
			_beta.setZero();
		}

		inline Eigen::VectorXd regress_single(const Eigen::VectorXd& input) const {
			return _beta.transpose() * input;
        }

		inline Eigen::VectorXd regress_multi(const Eigen::MatrixXd& input) const {
			return input * _beta;
		}

		inline size_t params_size() const {
			assert(_init);
			return _beta.size();
		}

		inline Eigen::VectorXd params() const {
			assert(_init);
			return Eigen::Map<const Eigen::VectorXd>(_beta.data(), _beta.size());
		}

		inline void set_params(const Eigen::VectorXd& beta) {
			assert(_init);
			_beta = Eigen::Map<const Eigen::MatrixXd>(beta.data(), _beta.rows(), _beta.cols());			
		}

		inline double cost(const Eigen::MatrixXd& input, const Eigen::MatrixXd& result) const {
			assert(_init);
			return _cost_internal(_beta, input, result);
		}

		inline double cost(const Eigen::VectorXd& beta, const Eigen::MatrixXd& input, const Eigen::MatrixXd& result) const {
			assert(_init);			
			return _cost_internal(Eigen::Map<const Eigen::MatrixXd>(beta.data(), _beta.rows(), _beta.cols()), input, result);
		}
		
		inline Eigen::VectorXd cost_gradient(const Eigen::MatrixXd& input, const Eigen::MatrixXd& result) const {
			assert(_init);
			Eigen::MatrixXd gradient = _cost_gradient_internal(_beta, input, result);
			return Eigen::Map<Eigen::VectorXd>(gradient.data(), gradient.size());
		}

		inline Eigen::VectorXd cost_gradient(const Eigen::VectorXd& beta, const Eigen::MatrixXd& input, const Eigen::MatrixXd& result) const {
			assert(_init);			
			Eigen::MatrixXd gradient = _cost_gradient_internal(Eigen::Map<const Eigen::MatrixXd>(beta.data(), _beta.rows(), _beta.cols()), input, result);
			return Eigen::Map<Eigen::VectorXd>(gradient.data(), gradient.size());
		}

		void self_train(const Eigen::MatrixXd& input, const Eigen::MatrixXd& result, bool reset = false) {
			// Closed-form solution of Least Squares Linear Regression: pinv(input' * input) * input' * result
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
						
			_beta = inv * input.transpose() * result;
			_init = true;
		}	

    protected:
		inline double _cost_internal(const Eigen::MatrixXd& beta, const Eigen::MatrixXd& input, const Eigen::MatrixXd& result) const {			
			return ((input * beta) - result).array().pow(2).sum() / (2 * result.rows());
		}

		inline Eigen::MatrixXd _cost_gradient_internal(const Eigen::MatrixXd& beta, const Eigen::MatrixXd& input, const Eigen::MatrixXd& result) const {
			return (input.transpose() * ((input * beta) - result)) / result.rows();
		}

		bool _init;
		Eigen::MatrixXd _beta;
	};
}
}
}
#endif