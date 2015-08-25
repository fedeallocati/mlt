#ifndef LINEAR_REGRESSOR_HPP
#define LINEAR_REGRESSOR_HPP

#include <Eigen/Core>
#include <Eigen/SVD>

namespace mlt {
namespace models {
namespace regressors {
    template <typename Params>
    class LinearRegressor {
    public:        
		typedef Params::LinearRegression params_t;
		typedef Eigen::Matrix<double, params_t::size + 1, 1> weights_t;
		typedef Eigen::Matrix<double, params_t::size + 1, 1> single_input_t;
		typedef Eigen::Matrix<double, 1, 1> single_result_t;
		typedef Eigen::Matrix<double, Eigen::Dynamic, params_t::size + 1> multiple_input_t;
		typedef Eigen::Matrix<double, Eigen::Dynamic, 1> multiple_result_t;

		void init() {
			_weights = weights_t::Zero();
		}

		single_result_t regress(single_input_t input) {			
			return _weights.transpose() * input;
        }

		double cost(multiple_input_t input, multiple_result_t result) {

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

			// Closed-form solution of Linear Regression
			_weights = (inv * input.transpose()) * result;
		}

		const weights_t& params() const {
			return _weights;
		}

    protected:
        weights_t _weights;
	};
}
}
}
#endif
