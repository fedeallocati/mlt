#ifndef MLT_MODELS_OPTIMIZABLE_LINEAR_MODEL_HPP
#define MLT_MODELS_OPTIMIZABLE_LINEAR_MODEL_HPP

#include <tuple>
#include <type_traits>

#include <Eigen/Core>

#include "../defs.hpp"
#include "../utils/eigen.hpp"
#include "../utils/linear_algebra.hpp"

namespace mlt {
namespace models {
	template <class LinearBase, class Loss, class Optimizer>
	class OptimizableLinearModel : public LinearBase {
	public:
		Self& fit(Features input, Target target, bool cold_start = true) {
			auto init = _fitted && !cold_start ? coefficients() : (MatrixXd::Random(target.rows(), input.rows() + (fit_intercept() ? 1 : 0)) * 0.005).eval();

			_set_coefficients(_optimizer(*this, input, _to_target_matrix(target), init, cold_start));

			return _self();
		}

		auto loss(MatrixXdRef coeffs, Features input, MatrixXdRef target) const {
			auto l = _loss.loss(_apply_linear_transformation(input, coeffs), target);
			if (_fit_intercept) {
				return l + _regularization * (coeffs.leftCols(coeffs.cols() - 1).array().pow(2).sum());
			}
			else {
				return l + _regularization * (coeffs.array().pow(2).sum());
			}
		}

		auto gradient(MatrixXdRef coeffs, Features input, MatrixXdRef target) const {
			auto g = _loss.gradient(_apply_linear_transformation(input, coeffs), target);

			if (_fit_intercept) {
				auto full_g = MatrixXd::Zero(coeffs.rows(), coeffs.cols()).eval();
				full_g.leftCols(coeffs.cols() - 1) = g * input.transpose() + _regularization * 2 * coeffs.leftCols(coeffs.cols() - 1);
				full_g.rightCols<1>() = g.rowwise().sum();
				return full_g;
			} else {
				return (g * input.transpose() + _regularization * 2 * coeffs).eval();
			}
		}

		auto loss_and_gradient(MatrixXdRef coeffs, Features input, MatrixXdRef target) const {
			double l;
			MatrixXd g;

			tie(l, g) = _loss.loss_and_gradient(_apply_linear_transformation(input, coeffs), target);
			if (_fit_intercept) {
				auto full_g = MatrixXd::Zero(coeffs.rows(), coeffs.cols()).eval();
				full_g.leftCols(coeffs.cols() - 1) = g * input.transpose() + _regularization * 2 * coeffs.leftCols(coeffs.cols() - 1);
				full_g.rightCols<1>() = g.rowwise().sum();
				return make_tuple(l + _regularization * (coeffs.leftCols(coeffs.cols() - 1).array().pow(2).sum()) , full_g);
			}
			else {
				return make_tuple(l + _regularization * coeffs.array().pow(2).sum(), (g * input.transpose() + _regularization * 2 * coeffs).eval());
			}
		}

	protected:
		template <class L, class O, class = enable_if<is_same<decay_t<L>, Loss>::value && is_convertible<decay_t<O>, Optimizer>::value>>
		explicit OptimizableLinearModel(L&& loss, O&& optimizer, double regularization, bool fit_intercept) : LinearBase(fit_intercept), _loss(forward<L>(loss)), _optimizer(forward<O>(optimizer)), _regularization(regularization) {}

		Loss _loss;
		Optimizer _optimizer;
		double _regularization;
	};
}
}
#endif	