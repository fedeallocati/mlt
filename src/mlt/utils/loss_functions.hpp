#ifndef MLT_UTILS_LOSS_FUNCTIONS_HPP
#define MLT_UTILS_LOSS_FUNCTIONS_HPP

#include <tuple>

#include <Eigen/Core>

#include "../defs.hpp"

namespace mlt {
namespace utils {
namespace loss_functions {
	class SquaredLoss {
	public:
		auto loss(MatrixXdRef pred, MatrixXdRef target) const {
			return (pred - target).array().pow(2).sum() / (2 * pred.cols());
		}

		auto gradient(MatrixXdRef pred, MatrixXdRef target) const {
			return MatrixXd{ (pred - target) / pred.cols() };
		}

		auto loss_and_gradient(MatrixXdRef pred, MatrixXdRef target) const {
			int size = pred.cols();
			auto residuals = (pred - target).eval();

			return make_tuple(residuals.array().pow(2).sum() / (2 * size), (residuals / size).eval());
		}
	};

	class HingeLoss {
	public:
		HingeLoss (double threshold = 1.0) : _threshold(threshold) {}

		auto loss(MatrixXdRef pred, MatrixXdRef target) const {
			return (((pred.rowwise() - pred.cwiseProduct(target).colwise().sum()) - target).array() + _threshold).max(0).sum() / pred.cols();
		}

		auto gradient(MatrixXdRef pred, MatrixXdRef target) const {
			auto margin_mask = ((((pred.rowwise() - pred.cwiseProduct(target).colwise().sum()) - target).array() + _threshold).max(0) > 0).cast<double>().eval();
			margin_mask = margin_mask + (target.array().rowwise() * -margin_mask.colwise().sum().array());
			return (margin_mask / pred.cols()).eval();
		}

		auto loss_and_gradient(MatrixXdRef pred, MatrixXdRef target) const {
			int size = pred.cols();
			auto hinge_loss = ((((pred.rowwise() - pred.cwiseProduct(target).colwise().sum()) - target).array() + _threshold).max(0)).eval();
			auto margin_mask = (hinge_loss.array() > 0).cast<double>().eval();
			margin_mask = margin_mask + (target.array().rowwise() * -margin_mask.colwise().sum().array());
			return make_tuple(hinge_loss.sum() / size, (margin_mask / size).eval());
		}
	protected:
		double _threshold;
	};

	class SoftmaxLoss {
	public:
		auto loss(MatrixXdRef pred, MatrixXdRef target) const {
			return -_softmax(pred).cwiseProduct(target).colwise().sum().array().log().sum() / pred.cols();
		}

		auto gradient(MatrixXdRef pred, MatrixXdRef target) const {
			return ((_softmax(pred) - target) / pred.cols()).eval();
		}

		auto loss_and_gradient(MatrixXdRef pred, MatrixXdRef target) const {
			int size = pred.cols();
			auto softmax_output = _softmax(pred);
			double l = softmax_output.cwiseProduct(target).colwise().sum().array().log().sum() / size;
			return make_tuple(l, ((_softmax(pred) - target) / size).eval());
		}

	protected:
		inline MatrixXd _softmax(MatrixXdRef x) const {
			auto result = (x.rowwise() - x.colwise().maxCoeff()).array().exp().eval();
			return (result.array().rowwise() / result.colwise().sum().array());
		}
	};
}
}
}
#endif