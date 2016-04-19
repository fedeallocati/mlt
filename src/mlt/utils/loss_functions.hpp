#ifndef MLT_UTILS_LOSS_FUNCTIONS_HPP
#define MLT_UTILS_LOSS_FUNCTIONS_HPP

#include <Eigen/Core>

namespace mlt {
namespace utils {
namespace loss_functions {
	class SquaredLoss {
	public:
		double loss(const Eigen::Ref<const Eigen::MatrixXd>& pred, const Eigen::Ref<const Eigen::MatrixXd>& target) const {
			return (pred - target).array().pow(2).sum() / (2 * pred.cols());
		}

		Eigen::MatrixXd gradient(const Eigen::Ref<const Eigen::MatrixXd>& pred, const Eigen::Ref<const Eigen::MatrixXd>& target) const {
			return (pred - target) / pred.cols();
		}

		std::tuple<double, Eigen::MatrixXd> loss_and_gradient(const Eigen::Ref<const Eigen::MatrixXd>& pred, const Eigen::Ref<const Eigen::MatrixXd>& target) const {
			return std::make_tuple(this->loss(pred, target), this->gradient(pred, target));
		}
	};

	class HingeLoss {
	public:
		HingeLoss (double threshold = 1.0) : _threshold(threshold) {}

		double loss(const Eigen::Ref<const Eigen::MatrixXd>& pred, const Eigen::Ref<const Eigen::MatrixXd>& target) const {
			return (((pred.rowwise() - pred.cwiseProduct(target).colwise().sum()) - target).array() + _threshold).max(0).sum() / pred.cols();
		}

		Eigen::MatrixXd gradient(const Eigen::Ref<const Eigen::MatrixXd>& pred, const Eigen::Ref<const Eigen::MatrixXd>& target) const {
			Eigen::MatrixXd margin_mask = ((((pred.rowwise() - pred.cwiseProduct(target).colwise().sum()) - target).array() + _threshold).max(0) > 0).cast<double>();
			return (margin_mask + (target.array().rowwise() * -margin_mask.colwise().sum().array()).matrix()) / pred.cols();
		}

		std::tuple<double, Eigen::MatrixXd> loss_and_gradient(const Eigen::Ref<const Eigen::MatrixXd>& pred, const Eigen::Ref<const Eigen::MatrixXd>& target) const {
			Eigen::MatrixXd hinge_loss = (((pred.rowwise() - pred.cwiseProduct(target).colwise().sum()) - target).array() + _threshold).max(0);
			Eigen::MatrixXd margin_mask = (hinge_loss.array() > 0).cast<double>();
			margin_mask = margin_mask + (target.array().rowwise() * -margin_mask.colwise().sum().array()).matrix();
			return std::make_tuple(hinge_loss.sum() / pred.cols(), margin_mask / pred.cols());
		}
	protected:
		double _threshold;
	};

	class SoftmaxLoss {
	public:
		double loss(const Eigen::Ref<const Eigen::MatrixXd>& pred, const Eigen::Ref<const Eigen::MatrixXd>& target) const {
			return -_softmax(pred).cwiseProduct(target).rowwise().sum().array().log().sum() / pred.cols();
		}

		Eigen::MatrixXd gradient(const Eigen::Ref<const Eigen::MatrixXd>& pred, const Eigen::Ref<const Eigen::MatrixXd>& target) const {
			return (_softmax(pred) - target) / pred.cols();
		}

		std::tuple<double, Eigen::MatrixXd> loss_and_gradient(const Eigen::Ref<const Eigen::MatrixXd>& pred, const Eigen::Ref<const Eigen::MatrixXd>& target) const {
			return std::make_tuple(this->loss(pred, target), this->gradient(pred, target));
		}
	protected:
		Eigen::MatrixXd _softmax(const Eigen::Ref<const Eigen::MatrixXd>& x) const {
			Eigen::MatrixXd result = (x.rowwise() - x.colwise().maxCoeff()).array().exp();
			return result.array().rowwise() / result.colwise().sum().array();
		}
	};
}
}
}
#endif