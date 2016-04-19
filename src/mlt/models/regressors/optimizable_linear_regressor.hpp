#ifndef MLT_MODELS_REGRESSORS_OPTIMIZABLE_LINEAR_REGRESSOR_HPP
#define MLT_MODELS_REGRESSORS_OPTIMIZABLE_LINEAR_REGRESSOR_HPP

#include <Eigen/Core>

#include "linear_regressor_model.hpp"
#include "../../utils/optimizers/stochastic_gradient_descent.hpp"

namespace mlt {
namespace models {
namespace regressors {
	template <class Concrete, class Optimizer>
	class OptimizableLinearModel {
	public:	
		template <class Target>
		Concrete& fit(const Eigen::MatrixXd& input, const Eigen::MatrixXd& target, bool cold_start) {
			Eigen::MatrixXd input_prime(input.rows() + (fit_intercept() ? 1 : 0), input.cols());
			input_prime.topRows(input.rows()) << input;

			if (static_cast<Concrete&>(*this).fit_intercept()) {
				input_prime.bottomRows<1>() = Eigen::VectorXd::Ones(input.cols());
			}

			Eigen::MatrixXd coeffs = _optimizer.run(static_cast<Concrete&>(*this), input_prime, target,
				(this->_fitted && !cold_start ? this->coefficients() : Eigen::MatrixXd::Random(target.rows(), input_prime.rows())), cold_start);

			if (static_cast<Concrete&>(*this).fit_intercept()) {
				static_cast<Concrete&>(*this)._set_coefficients_and_intercepts(coeffs.leftCols(coeffs.cols() - 1), coeffs.rightCols<1>());
			}
			else {
				static_cast<Concrete&>(*this)._set_coefficients(coeffs);
			}

			return static_cast<Concrete&>(*this);
		}

		double loss(const Eigen::Ref<const Eigen::MatrixXd>& params, const Eigen::MatrixXd& input, const Eigen::MatrixXd& result) const {
			return (((params * input) - result).array().pow(2).sum() / (2 * input.cols())) + _reg * (params.array().pow(2)).sum();
		}

		std::tuple<double, Eigen::MatrixXd> loss_and_gradient(const Eigen::Ref<const Eigen::MatrixXd>& params, const Eigen::MatrixXd& input, const Eigen::MatrixXd& result) const {
			Eigen::MatrixXd diff = params * input - result;
			return std::make_tuple((diff.array().pow(2).sum() / (2 * input.cols())) + 0.5 * (params.array().pow(2)).sum(),
				((diff * input.transpose()) / input.cols()) + _reg * 2 * params);
		}

	protected:
		explicit OptimizableLinearModel(const Optimizer& optimizer,
			bool fit_intercept = true) : LinearRegressorModel(fit_intercept), _loss(loss), _optimizer(optimizer) {}

		explicit OptimizableLinearModel(Optimizer&& optimizer,
			bool fit_intercept = true) : LinearRegressorModel(fit_intercept), _optimizer(optimizer) {}

		double _regularization;
		Optimizer _optimizer;
	};
}
}
}
#endif