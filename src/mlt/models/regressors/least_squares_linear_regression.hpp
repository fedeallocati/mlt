#ifndef MLT_MODELS_REGRESSORS_LEAST_SQUARES_LINEAR_REGRESSION_HPP
#define MLT_MODELS_REGRESSORS_LEAST_SQUARES_LINEAR_REGRESSION_HPP

#include <Eigen/Core>

#include "linear_regressor.hpp"
#include "../../utils/linear_solvers.hpp"

namespace mlt {
namespace models {
namespace regressors {
	template <class Solver = utils::linear_solvers::SVDSolver>
    class LeastSquaresLinearRegression : public LinearRegressor<LeastSquaresLinearRegression<Solver>> {
    public:         
        explicit LeastSquaresLinearRegression(bool fit_intercept = true) : LinearRegressor(fit_intercept), _solver(Solver()) {}

		explicit LeastSquaresLinearRegression(const Solver& solver, bool fit_intercept = true) : LinearRegressor(fit_intercept), _solver(solver) {}

		explicit LeastSquaresLinearRegression(Solver&& solver, bool fit_intercept = true) : LinearRegressor(fit_intercept), _solver(solver) {}

        LeastSquaresLinearRegression& fit(const Eigen::MatrixXd& input, const Eigen::MatrixXd& target, bool = true) {
            Eigen::MatrixXd input_prime(input.rows() + (_fit_intercept ? 1 : 0), input.cols());
			input_prime.topRows(input.rows()) << input;

			if (_fit_intercept) {
				input_prime.bottomRows<1>() = Eigen::VectorXd::Ones(input.cols());
			}

			_set_coefficients(_solver.compute(input_prime * input_prime.transpose()).solve(input_prime * target.transpose()).transpose());

			return *this;
        }
	protected:
		Solver _solver;
    };
}
}
}
#endif