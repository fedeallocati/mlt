#ifndef MLT_MODELS_OPTIMIZABLE_LINEAR_MODEL_HPP
#define MLT_MODELS_OPTIMIZABLE_LINEAR_MODEL_HPP

#include <Eigen/Core>

#include "../utils/linear_algebra.hpp"

namespace mlt {
namespace models {
	template <class LinearBase, class Loss, class Optimizer>
	class OptimizableLinearModel : public LinearBase {
	public:
		double loss(const Eigen::Ref<const Eigen::MatrixXd>& coeffs, const Eigen::Ref<const Eigen::MatrixXd>& input, const Eigen::Ref<const Eigen::MatrixXd>& target) const {
			if (this->_fit_intercept) {
				return _loss.loss(utils::linear_algebra::linear_transformation(input, coeffs.leftCols(coeffs.cols() - 1), coeffs.rightCols<1>()), target) +
					_regularization * (coeffs.leftCols(coeffs.cols() - 1).array().pow(2).sum());
			}
			else {
				return _loss.loss(utils::linear_algebra::linear_transformation(input, coeffs), target) + _regularization * (coeffs.array().pow(2).sum());
			}
		}

		Eigen::MatrixXd gradient(const Eigen::Ref<const Eigen::MatrixXd>& coeffs, const Eigen::Ref<const Eigen::MatrixXd>& input, const Eigen::Ref<const Eigen::MatrixXd>& target) const {
			if (this->_fit_intercept) {
				Eigen::MatrixXd loss_grad = _loss.gradient(utils::linear_algebra::linear_transformation(input, coeffs.leftCols(coeffs.cols() - 1), coeffs.rightCols<1>()), target);
				Eigen::MatrixXd gradient = Eigen::MatrixXd::Zero(coeffs.rows(), coeffs.cols());
				gradient.leftCols(coeffs.cols() - 1) = loss_grad * input.transpose() + _regularization * 2 * coeffs.leftCols(coeffs.cols() - 1);
				gradient.rightCols<1>() = loss_grad.rowwise().sum();
				return gradient;
			}
			else {
				return _loss.gradient(utils::linear_algebra::linear_transformation(input, coeffs), target) * input.transpose() + _regularization * 2 * coeffs;
			}
		}

		std::tuple<double, Eigen::MatrixXd> loss_and_gradient(const Eigen::Ref<const Eigen::MatrixXd>& coeffs, const Eigen::Ref<const Eigen::MatrixXd>& input, const Eigen::Ref<const Eigen::MatrixXd>& target) const {
			if (this->_fit_intercept) {
				std::tuple<double, Eigen::MatrixXd> res = _loss.loss_and_gradient(utils::linear_algebra::linear_transformation(input, coeffs.leftCols(coeffs.cols() - 1), coeffs.rightCols<1>()), target);
				Eigen::MatrixXd gradient = Eigen::MatrixXd::Zero(coeffs.rows(), coeffs.cols());
				gradient.leftCols(coeffs.cols() - 1) = std::get<1>(res) * input.transpose() + _regularization * 2 * coeffs.leftCols(coeffs.cols() - 1);
				gradient.rightCols<1>() = std::get<1>(res).rowwise().sum();
				return { std::get<0>(res) + _regularization * (coeffs.leftCols(coeffs.cols() - 1).array().pow(2).sum()), gradient };
			}
			else {
				std::tuple<double, Eigen::MatrixXd> res = _loss.loss_and_gradient(utils::linear_algebra::linear_transformation(input, coeffs), target);
				return { std::get<0>(res) + _regularization * coeffs.array().pow(2).sum(), std::get<1>(res) * input.transpose() + _regularization * 2 * coeffs };
			}
		}

	protected:
		template <class L, class O, class = std::enable_if<std::is_same<std::decay_t<L>, Loss>::value && std::is_convertible<std::decay_t<O>, Optimizer>::value>>
		explicit OptimizableLinearModel(L&& loss, O&& optimizer, double regularization, bool fit_intercept) :
			LinearBase(fit_intercept), _loss(std::forward<L>(loss)), _optimizer(std::forward<O>(optimizer)), _regularization(regularization) {}

		void _fit(const Eigen::Ref<const Eigen::MatrixXd>& input, const Eigen::Ref<const Eigen::MatrixXd>& target, bool cold_start = true) {
			Eigen::MatrixXd init = this->_fitted && !cold_start ?
				this->coefficients() :
				(Eigen::MatrixXd::Random(target.rows(), input.rows() + (this->fit_intercept() ? 1 : 0)) * 0.005);

			this->_set_coefficients(_optimizer.run(*this, input, target, init, cold_start));
		}

		Loss _loss;
		Optimizer _optimizer;
		double _regularization;
	};
}
}
#endif	