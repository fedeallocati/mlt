#ifndef MLT_MODELS_REGRESSORS_LEAST_SQUARES_LINEAR_REGRESSION_HPP
#define MLT_MODELS_REGRESSORS_LEAST_SQUARES_LINEAR_REGRESSION_HPP

#include <type_traits>

#include <Eigen/Core>

#include "linear_regressor.hpp"
#include "../../utils/linear_solvers.hpp"

namespace mlt {
namespace models {
namespace regressors {
	using namespace utils::linear_solvers;

	template <class Solver = SVDSolver>
    class LeastSquaresLinearRegression : public LinearRegressor<LeastSquaresLinearRegression<Solver>> {
    public:         
        explicit LeastSquaresLinearRegression(bool fit_intercept = true) : LinearRegressor(fit_intercept), _solver(Solver()) {}

        template <class S, class = enable_if<is_same<decay_t<S>, Solver>::value>>
		explicit LeastSquaresLinearRegression(const S&& solver, bool fit_intercept = true) : LinearRegressor(fit_intercept), _solver(forward<S>(solver)) {}

        Self& fit(Features input, Target target, bool = true) {
            MatrixXd input_prime(input.rows() + (_fit_intercept ? 1 : 0), input.cols());
			input_prime.topRows(input.rows()) << input;

			if (_fit_intercept) {
				input_prime.bottomRows<1>() = VectorXd::Ones(input.cols());
			}

			_set_coefficients(_solver.compute(input_prime * input_prime.transpose()).solve(input_prime * target.transpose()).transpose());

			return _self();
        }

	protected:
		Solver _solver;
    };
}
}
}
#endif