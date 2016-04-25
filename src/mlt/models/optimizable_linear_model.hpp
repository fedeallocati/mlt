#ifndef MLT_MODELS_OPTIMIZABLE_LINEAR_MODEL_HPP
#define MLT_MODELS_OPTIMIZABLE_LINEAR_MODEL_HPP

#include <Eigen/Core>

#include "linear_model.hpp"

namespace mlt {
namespace models {
	template <class Loss, class Optimizer>
	class OptimizableLinearModel : public LinearModel {
	public:
		explicit OptimizableLinearModel(const Loss& loss, const Optimizer& optimizer, double regularization,
			bool fit_intercept = true) : LinearModel(fit_intercept), _loss(loss), _optimizer(optimizer),
			_regularization(regularization) {}

		explicit OptimizableLinearModel(Loss&& loss, const Optimizer& optimizer, double regularization,
			bool fit_intercept = true) : LinearModel(fit_intercept), _loss(loss), _optimizer(optimizer),
			_regularization(regularization) {}

		explicit OptimizableLinearModel(const Loss& loss, Optimizer&& optimizer, double regularization,
			bool fit_intercept = true) : LinearModel(fit_intercept), _loss(loss), _optimizer(optimizer),
			_regularization(regularization) {}

		explicit OptimizableLinearModel(Loss&& loss, Optimizer&& optimizer, double regularization,
			bool fit_intercept = true) : LinearModel(fit_intercept), _loss(loss), _optimizer(optimizer),
			_regularization(regularization) {}

		OptimizableLinearModel& fit(const Eigen::Ref<const Eigen::MatrixXd>& input, const Eigen::Ref<const Eigen::MatrixXd>& target, bool cold_start = true) {
			Eigen::MatrixXd init = this->_fitted && !cold_start ?
				this->coefficients() : 
				(Eigen::MatrixXd::Random(target.rows(), input.rows() + (_fit_intercept ? 1 : 0)) * 0.005);

			_set_coefficients(_optimizer.run(*this, input, target, init, cold_start));
			return *this;
		}

		Eigen::MatrixXd predict(const Eigen::MatrixXd& input) const {
			assert(_fitted);
			return _apply_linear_transformation(input);
		}

		double loss(const Eigen::Ref<const Eigen::MatrixXd>& coeffs, const Eigen::Ref<const Eigen::MatrixXd>& input, const Eigen::Ref<const Eigen::MatrixXd>& target) const {
			if (_fit_intercept) {
				return _loss.loss(_apply_linear_transformation(input, coeffs.leftCols(coeffs.cols() - 1), coeffs.rightCols<1>()), target) +
					_regularization * (coeffs.leftCols(coeffs.cols() - 1).array().pow(2).sum());
			}
			else {
				return _loss.loss(_apply_linear_transformation(input, coeffs), target) + _regularization * (coeffs.array().pow(2).sum());
			}
		}

		Eigen::MatrixXd gradient(const Eigen::Ref<const Eigen::MatrixXd>& coeffs, const Eigen::Ref<const Eigen::MatrixXd>& input, const Eigen::Ref<const Eigen::MatrixXd>& target) const {
			if (_fit_intercept) {
				Eigen::MatrixXd loss_grad = _loss.gradient(_apply_linear_transformation(input, coeffs.leftCols(coeffs.cols() - 1), coeffs.rightCols<1>()), target);
				Eigen::MatrixXd gradient = Eigen::MatrixXd::Zero(coeffs.rows(), coeffs.cols());
				gradient.leftCols(coeffs.cols() - 1) = loss_grad * input.transpose() + _regularization * 2 * coeffs.leftCols(coeffs.cols() - 1);
				gradient.rightCols<1>() = loss_grad.rowwise().sum();
				return gradient;
			}
			else {
				return _loss.gradient(_apply_linear_transformation(input, coeffs), target) * input.transpose() + _regularization * 2 * coeffs;
			}
		}

		std::tuple<double, Eigen::MatrixXd> loss_and_gradient(const Eigen::Ref<const Eigen::MatrixXd>& coeffs, const Eigen::Ref<const Eigen::MatrixXd>& input, const Eigen::Ref<const Eigen::MatrixXd>& target) const {
			if (_fit_intercept) {
				std::tuple<double, Eigen::MatrixXd> res = _loss.loss_and_gradient(_apply_linear_transformation(input, coeffs.leftCols(coeffs.cols() - 1), coeffs.rightCols<1>()), target);
				Eigen::MatrixXd gradient = Eigen::MatrixXd::Zero(coeffs.rows(), coeffs.cols());
				gradient.leftCols(coeffs.cols() - 1) = std::get<1>(res) * input.transpose() + _regularization * 2 * coeffs.leftCols(coeffs.cols() - 1);
				gradient.rightCols<1>() = std::get<1>(res).rowwise().sum();
				return { std::get<0>(res) + _regularization * (coeffs.leftCols(coeffs.cols() - 1).array().pow(2).sum()), gradient };
			}
			else {
				std::tuple<double, Eigen::MatrixXd> res = _loss.loss_and_gradient(_apply_linear_transformation(input, coeffs), target);
				return { std::get<0>(res) + _regularization * coeffs.array().pow(2).sum(), std::get<1>(res) * input.transpose() + _regularization * 2 * coeffs };
			}
		}

	protected:
		Loss _loss;
		Optimizer _optimizer;
		double _regularization;
	};
}
}
#endif