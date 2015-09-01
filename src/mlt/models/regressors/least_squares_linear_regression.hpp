#ifndef LEAST_SQUARES_LINEAR_REGRESSION_HPP
#define LEAST_SQUARES_LINEAR_REGRESSION_HPP

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
    // - Supervision: Supervised
	// Parameters:
	// - double regularization: amount of L2 regularization to apply. Set to 0 or less if don't want to use.	
	template <typename Params>
    class LeastSquaresLinearRegression {
    public:         
        LeastSquaresLinearRegression() : _init(false) {}

        LeastSquaresLinearRegression(size_t input, size_t output) : _init(true), _beta(Eigen::MatrixXd::Zero(input + 1, output)) {}

        // Disable copy constructors
        LeastSquaresLinearRegression(const LeastSquaresLinearRegression& other) = delete;
        LeastSquaresLinearRegression& operator=(const LeastSquaresLinearRegression& other) = delete;

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
            assert(_init);
            return _beta.transpose() * input;
        }

        inline Eigen::MatrixXd regress_multi(const Eigen::MatrixXd& input) const {
            assert(_init);
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
        
        inline std::tuple<double, Eigen::VectorXd> cost_and_gradient(const Eigen::MatrixXd& input, const Eigen::MatrixXd& result) const {
            assert(_init);
            auto c_a_g = _cost_and_gradient_internal(_beta, input, result);
            return std::make_tuple(std::get<0>(c_a_g), Eigen::Map<Eigen::VectorXd>(std::get<1>(c_a_g).data(), std::get<1>(c_a_g).size()));
        }

        inline std::tuple<double, Eigen::VectorXd> cost_and_gradient(const Eigen::VectorXd& beta, const Eigen::MatrixXd& input, const Eigen::MatrixXd& result) const {
            assert(_init);
            auto c_a_g = _cost_and_gradient_internal(Eigen::Map<const Eigen::MatrixXd>(beta.data(), _beta.rows(), _beta.cols()), input, result);
            return std::make_tuple(std::get<0>(c_a_g), Eigen::Map<Eigen::VectorXd>(std::get<1>(c_a_g).data(), std::get<1>(c_a_g).size()));
        }

        void self_train(const Eigen::MatrixXd& input, const Eigen::MatrixXd& result, bool reset = false) {
            // Closed-form solution of Least Squares Linear Regression: pinv(input' * input) * input' * result
            // Moore-Penrose pseudoinverse
            double epsilon = 1e-9;
			Eigen::MatrixXd matrix_to_inverse = input.transpose() * input;
			if (params_t::regularization > 0) {
				matrix_to_inverse += Eigen::MatrixXd::Identity(matrix_to_inverse.rows(), matrix_to_inverse.cols()) * params_t::regularization;
			}			
            Eigen::JacobiSVD<Eigen::MatrixXd> svd(matrix_to_inverse, Eigen::ComputeThinU | Eigen::ComputeThinV);
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
		typedef Params::LeastSquaresLinearRegression params_t;

        inline double _cost_internal(const Eigen::MatrixXd& beta, const Eigen::MatrixXd& input, const Eigen::MatrixXd& result) const {          
			double loss = ((input * beta) - result).array().pow(2).sum() / (2 * input.rows());
			if (params_t::regularization > 0) {
				loss += params_t::regularization * (beta.array().pow(2)).sum();
			}
            return loss;
        }

        inline std::tuple<double, Eigen::MatrixXd> _cost_and_gradient_internal(const Eigen::MatrixXd& beta, const Eigen::MatrixXd& input, const Eigen::MatrixXd& result) const {
            Eigen::MatrixXd diff = input * beta - result;
            double loss = diff.array().pow(2).sum() / (2 * input.rows());
			if (params_t::regularization > 0) {
				loss += params_t::regularization * (beta.array().pow(2)).sum();
			}
            Eigen::MatrixXd d_beta = (input.transpose() * diff) / input.rows();
			if (params_t::regularization > 0) {
				d_beta += params_t::regularization * 2 * beta;
			}
            return std::make_tuple(loss, d_beta);
        }

        bool _init;
        Eigen::MatrixXd _beta;
    };
}
}
}
#endif