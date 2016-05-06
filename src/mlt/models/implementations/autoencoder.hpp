#ifndef MLT_MODELS_IMPLEMENTATIONS_AUTOENCODER_HPP
#define MLT_MODELS_IMPLEMENTATIONS_AUTOENCODER_HPP

#include <Eigen/Core>

namespace mlt {
namespace models {
namespace implementations {
namespace autoencoder
{
	template <class HiddenActivation, class ReconstructionActivation>
	double loss(const HiddenActivation& hidden_activation, const ReconstructionActivation& reconstruction_activation,
	const Eigen::Ref<const Eigen::MatrixXd>& hidden_weights, const Eigen::Ref<const Eigen::VectorXd>& hidden_intercepts,
	const Eigen::Ref<const Eigen::MatrixXd>& reconstruction_weights, const Eigen::Ref<const Eigen::VectorXd>& reconstruction_intercepts,
	double regularization, const Eigen::Ref<const Eigen::MatrixXd>& input, const Eigen::Ref<const Eigen::MatrixXd>& target) {
		auto hidden_z = Eigen::MatrixXd{ (hidden_weights * input).colwise() + hidden_intercepts };
		auto hidden_a = Eigen::MatrixXd{ hidden_activation.compute(hidden_z) };
		auto reconstruction_z = Eigen::MatrixXd{ (reconstruction_weights * hidden_a).colwise() + reconstruction_intercepts };

		return (((reconstruction_activation.compute(reconstruction_z)) - target).array().pow(2).sum() / (2 * input.cols())) +
			regularization * hidden_weights.array().pow(2).sum() +
			regularization * reconstruction_weights.array().pow(2).sum();
	}

	template <class HiddenActivation, class ReconstructionActivation>
	std::tuple<Eigen::MatrixXd, Eigen::VectorXd, Eigen::MatrixXd, Eigen::VectorXd> 
	gradient(const HiddenActivation& hidden_activation, const ReconstructionActivation& reconstruction_activation,
	const Eigen::Ref<const Eigen::MatrixXd>& hidden_weights, const Eigen::Ref<const Eigen::VectorXd>& hidden_intercepts,
	const Eigen::Ref<const Eigen::MatrixXd>& reconstruction_weights, const Eigen::Ref<const Eigen::VectorXd>& reconstruction_intercepts,
	double regularization, const Eigen::Ref<const Eigen::MatrixXd>& input, const Eigen::Ref<const Eigen::MatrixXd>& target) {
		auto hidden_z = Eigen::MatrixXd{ (hidden_weights * input).colwise() + hidden_intercepts };
		auto hidden_a = Eigen::MatrixXd{ hidden_activation.compute(hidden_z) };
		auto reconstruction_z = Eigen::MatrixXd{ (reconstruction_weights * hidden_a).colwise() + reconstruction_intercepts };
		auto recontstruction_error = Eigen::MatrixXd{ (reconstruction_activation.compute(reconstruction_z) - target) };
		auto reconstruction_delta = Eigen::MatrixXd{ recontstruction_error.cwiseProduct(reconstruction_activation.gradient(reconstruction_z)) };

		auto hidden_delta = Eigen::MatrixXd{ (reconstruction_weights.transpose() * reconstruction_delta).cwiseProduct(hidden_activation.gradient(hidden_z)) };

		return{ (hidden_delta * input.transpose() / input.cols()) + regularization * 2 * hidden_weights,
			hidden_delta.rowwise().sum() / input.cols(),
			(reconstruction_delta * hidden_a.transpose() / input.cols()) + regularization * 2 * reconstruction_weights,
			reconstruction_delta.rowwise().sum() / input.cols() };
	}

	template <class HiddenActivation, class ReconstructionActivation>
	std::tuple<double, Eigen::MatrixXd, Eigen::VectorXd, Eigen::MatrixXd, Eigen::VectorXd>
	loss_and_gradient(const HiddenActivation& hidden_activation, const ReconstructionActivation& reconstruction_activation,
	const Eigen::Ref<const Eigen::MatrixXd>& hidden_weights, const Eigen::Ref<const Eigen::VectorXd>& hidden_intercepts,
	const Eigen::Ref<const Eigen::MatrixXd>& reconstruction_weights, const Eigen::Ref<const Eigen::VectorXd>& reconstruction_intercepts,
	double regularization, const Eigen::Ref<const Eigen::MatrixXd>& input, const Eigen::Ref<const Eigen::MatrixXd>& target) {
		auto hidden_z = Eigen::MatrixXd{ (hidden_weights * input).colwise() + hidden_intercepts };
		auto hidden_a = Eigen::MatrixXd{ hidden_activation.compute(hidden_z) };
		auto reconstruction_z = Eigen::MatrixXd{ (reconstruction_weights * hidden_a).colwise() + reconstruction_intercepts };
		auto recontstruction_error = Eigen::MatrixXd{ (reconstruction_activation.compute(reconstruction_z) - target) };
		auto reconstruction_delta = Eigen::MatrixXd{ recontstruction_error.cwiseProduct(reconstruction_activation.gradient(reconstruction_z)) };

		auto loss = (recontstruction_error.array().pow(2).sum() / (2 * input.cols())) +
			regularization * hidden_weights.array().pow(2).sum() +
			regularization * reconstruction_weights.array().pow(2).sum();

		auto hidden_delta = Eigen::MatrixXd{ (reconstruction_weights.transpose() * reconstruction_delta).cwiseProduct(hidden_activation.gradient(hidden_z)) };

		return{ loss,
			(hidden_delta * input.transpose() / input.cols()) + regularization * 2 * hidden_weights,
			hidden_delta.rowwise().sum() / input.cols(),
			(reconstruction_delta * hidden_a.transpose() / input.cols()) + regularization * 2 * reconstruction_weights,
			reconstruction_delta.rowwise().sum() / input.cols() };
	}

	template <class HiddenActivation, class ReconstructionActivation>
	double sparse_loss(const HiddenActivation& hidden_activation, const ReconstructionActivation& reconstruction_activation,
	const Eigen::Ref<const Eigen::MatrixXd>& hidden_weights, const Eigen::Ref<const Eigen::VectorXd>& hidden_intercepts,
	const Eigen::Ref<const Eigen::MatrixXd>& reconstruction_weights, const Eigen::Ref<const Eigen::VectorXd>& reconstruction_intercepts,
	double regularization, double sparsity, double sparsity_weight,
	const Eigen::Ref<const Eigen::MatrixXd>& input, const Eigen::Ref<const Eigen::MatrixXd>& target) {
		auto hidden_z = Eigen::MatrixXd{ (hidden_weights * input).colwise() + hidden_intercepts };
		auto hidden_a = Eigen::MatrixXd{ hidden_activation.compute(hidden_z) };
		auto reconstruction_z = Eigen::MatrixXd{ (reconstruction_weights * hidden_a).colwise() + reconstruction_intercepts };

		auto rho_hat = Eigen::MatrixXd{ (hidden_a.rowwise().sum() / input.cols()).unaryExpr([](double x) { return std::abs(x - 1.0) < std::numeric_limits<double>::epsilon() ? (x + std::numeric_limits<double>::epsilon()) : x; }) };
		auto sparsity_penalty = ((sparsity * (sparsity / rho_hat.array()).log()) + ((1 - sparsity) * ((1 - sparsity) / (1 - rho_hat.array())).log())).sum();

		return (((reconstruction_activation.compute(reconstruction_z)) - target).array().pow(2).sum() / (2 * input.cols())) +
			regularization * hidden_weights.array().pow(2).sum() +
			regularization * reconstruction_weights.array().pow(2).sum() +
			sparsity_weight * sparsity_penalty;
	}

	template <class HiddenActivation, class ReconstructionActivation>
	std::tuple<Eigen::MatrixXd, Eigen::VectorXd, Eigen::MatrixXd, Eigen::VectorXd>
	sparse_gradient(const HiddenActivation& hidden_activation, const ReconstructionActivation& reconstruction_activation,
	const Eigen::Ref<const Eigen::MatrixXd>& hidden_weights, const Eigen::Ref<const Eigen::VectorXd>& hidden_intercepts,
	const Eigen::Ref<const Eigen::MatrixXd>& reconstruction_weights, const Eigen::Ref<const Eigen::VectorXd>& reconstruction_intercepts,
	double regularization, double sparsity, double sparsity_weight,
	const Eigen::Ref<const Eigen::MatrixXd>& input, const Eigen::Ref<const Eigen::MatrixXd>& target) {
		auto hidden_z = Eigen::MatrixXd{ (hidden_weights * input).colwise() + hidden_intercepts };
		auto hidden_a = Eigen::MatrixXd{ hidden_activation.compute(hidden_z) };
		auto reconstruction_z = Eigen::MatrixXd{ (reconstruction_weights * hidden_a).colwise() + reconstruction_intercepts };
		auto recontstruction_error = Eigen::MatrixXd{ (reconstruction_activation.compute(reconstruction_z) - target) };
		auto reconstruction_delta = Eigen::MatrixXd{ recontstruction_error.cwiseProduct(reconstruction_activation.gradient(reconstruction_z)) };

		auto rho_hat = Eigen::MatrixXd{ (hidden_a.rowwise().sum() / input.cols()).unaryExpr([](double x) { return std::abs(x - 1.0) < std::numeric_limits<double>::epsilon() ? (x + std::numeric_limits<double>::epsilon()) : x; }) };
		auto sparsity_delta = Eigen::VectorXd{ (-sparsity / rho_hat.array()) + ((1 - sparsity) / (1 - rho_hat.array())) };
		auto hidden_delta = Eigen::MatrixXd{ ((reconstruction_weights.transpose() * reconstruction_delta).colwise() +
			(sparsity_weight * sparsity_delta)).cwiseProduct(hidden_activation.gradient(hidden_z)) };

		return{ (hidden_delta * input.transpose() / input.cols()) + regularization * 2 * hidden_weights,
			hidden_delta.rowwise().sum() / input.cols(),
			(reconstruction_delta * hidden_a.transpose() / input.cols()) + regularization * 2 * reconstruction_weights,
			reconstruction_delta.rowwise().sum() / input.cols() };
	}

	template <class HiddenActivation, class ReconstructionActivation>
	std::tuple<double, Eigen::MatrixXd, Eigen::VectorXd, Eigen::MatrixXd, Eigen::VectorXd>
	sparse_loss_and_gradient(const HiddenActivation& hidden_activation, const ReconstructionActivation& reconstruction_activation,
	const Eigen::Ref<const Eigen::MatrixXd>& hidden_weights, const Eigen::Ref<const Eigen::VectorXd>& hidden_intercepts,
	const Eigen::Ref<const Eigen::MatrixXd>& reconstruction_weights, const Eigen::Ref<const Eigen::VectorXd>& reconstruction_intercepts,
	double regularization, double sparsity, double sparsity_weight,
	const Eigen::Ref<const Eigen::MatrixXd>& input, const Eigen::Ref<const Eigen::MatrixXd>& target) {
		auto hidden_z = Eigen::MatrixXd{ (hidden_weights * input).colwise() + hidden_intercepts };
		auto hidden_a = Eigen::MatrixXd{ hidden_activation.compute(hidden_z) };
		auto reconstruction_z = Eigen::MatrixXd{ (reconstruction_weights * hidden_a).colwise() + reconstruction_intercepts };
		auto recontstruction_error = Eigen::MatrixXd{ (reconstruction_activation.compute(reconstruction_z) - target) };
		auto reconstruction_delta = Eigen::MatrixXd{ recontstruction_error.cwiseProduct(reconstruction_activation.gradient(reconstruction_z)) };

		auto rho_hat = Eigen::MatrixXd{ (hidden_a.rowwise().sum() / input.cols()).unaryExpr([](double x) { return std::abs(x - 1.0) < std::numeric_limits<double>::epsilon() ? (x + std::numeric_limits<double>::epsilon()) : x; }) };
		auto sparsity_delta = Eigen::VectorXd{ (-sparsity / rho_hat.array()) + ((1 - sparsity) / (1 - rho_hat.array())) };
		auto hidden_delta = Eigen::MatrixXd{ ((reconstruction_weights.transpose() * reconstruction_delta).colwise() +
			(sparsity_weight * sparsity_delta)).cwiseProduct(hidden_activation.gradient(hidden_z)) };

		auto loss = (recontstruction_error.array().pow(2).sum() / (2 * input.cols())) +
			regularization * hidden_weights.array().pow(2).sum() +
			regularization * reconstruction_weights.array().pow(2).sum() +
			sparsity_weight * ((sparsity * (sparsity / rho_hat.array()).log()) + ((1 - sparsity) * ((1 - sparsity) / (1 - rho_hat.array())).log())).sum();

		return{ loss,
			(hidden_delta * input.transpose() / input.cols()) + regularization * 2 * hidden_weights,
			hidden_delta.rowwise().sum() / input.cols(),
			(reconstruction_delta * hidden_a.transpose() / input.cols()) + regularization * 2 * reconstruction_weights,
			reconstruction_delta.rowwise().sum() / input.cols() };
	}
}
}
}
}
#endif