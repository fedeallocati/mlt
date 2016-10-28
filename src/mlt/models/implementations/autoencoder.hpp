#ifndef MLT_MODELS_IMPLEMENTATIONS_AUTOENCODER_HPP
#define MLT_MODELS_IMPLEMENTATIONS_AUTOENCODER_HPP

#include <limits>
#include <tuple>

#include <Eigen/Core>

#include "../../defs.hpp"

namespace mlt {
namespace models {
namespace implementations {
namespace autoencoder {
	template <class HiddenActivation, class ReconstructionActivation>
	auto loss(const HiddenActivation& hidden_activation, const ReconstructionActivation& reconstruction_activation,
	MatrixXdRef hidden_weights, VectorXdRef hidden_intercepts,
	MatrixXdRef reconstruction_weights, VectorXdRef reconstruction_intercepts,
	double regularization, MatrixXdRef input, MatrixXdRef target) {
		auto hidden_z = ((hidden_weights * input).colwise() + hidden_intercepts).eval();
		auto hidden_a = (hidden_activation.compute(hidden_z)).eval();
		auto reconstruction_z = ((reconstruction_weights * hidden_a).colwise() + reconstruction_intercepts).eval();

		return (((reconstruction_activation.compute(reconstruction_z)) - target).array().pow(2).sum() / (2 * input.cols())) +
			regularization * hidden_weights.array().pow(2).sum() +
			regularization * reconstruction_weights.array().pow(2).sum();
	}

	template <class HiddenActivation, class ReconstructionActivation>
	tuple<MatrixXd, VectorXd, MatrixXd, VectorXd> 
	gradient(const HiddenActivation& hidden_activation, const ReconstructionActivation& reconstruction_activation,
	MatrixXdRef hidden_weights, VectorXdRef hidden_intercepts,
	MatrixXdRef reconstruction_weights, VectorXdRef reconstruction_intercepts,
	double regularization, MatrixXdRef input, MatrixXdRef target) {
		auto hidden_z = ((hidden_weights * input).colwise() + hidden_intercepts).eval();
		auto hidden_a = (hidden_activation.compute(hidden_z)).eval();
		auto reconstruction_z = ((reconstruction_weights * hidden_a).colwise() + reconstruction_intercepts).eval();
		auto recontstruction_error = ((reconstruction_activation.compute(reconstruction_z) - target)).eval();
		auto reconstruction_delta = (recontstruction_error.cwiseProduct(reconstruction_activation.gradient(reconstruction_z))).eval();

		auto hidden_delta = ((reconstruction_weights.transpose() * reconstruction_delta).cwiseProduct(hidden_activation.gradient(hidden_z))).eval();

		return{ (hidden_delta * input.transpose() / input.cols()) + regularization * 2 * hidden_weights,
			hidden_delta.rowwise().sum() / input.cols(),
			(reconstruction_delta * hidden_a.transpose() / input.cols()) + regularization * 2 * reconstruction_weights,
			reconstruction_delta.rowwise().sum() / input.cols() };
	}

	template <class HiddenActivation, class ReconstructionActivation>
	tuple<double, MatrixXd, VectorXd, MatrixXd, VectorXd>
	loss_and_gradient(const HiddenActivation& hidden_activation, const ReconstructionActivation& reconstruction_activation,
	MatrixXdRef hidden_weights, VectorXdRef hidden_intercepts,
	MatrixXdRef reconstruction_weights, VectorXdRef reconstruction_intercepts,
	double regularization, MatrixXdRef input, MatrixXdRef target) {
		auto hidden_z = ((hidden_weights * input).colwise() + hidden_intercepts).eval();
		auto hidden_a = (hidden_activation.compute(hidden_z)).eval();
		auto reconstruction_z = ((reconstruction_weights * hidden_a).colwise() + reconstruction_intercepts).eval();
		auto recontstruction_error = ((reconstruction_activation.compute(reconstruction_z) - target)).eval();
		auto reconstruction_delta = (recontstruction_error.cwiseProduct(reconstruction_activation.gradient(reconstruction_z))).eval();

		auto loss = (recontstruction_error.array().pow(2).sum() / (2 * input.cols())) +
			regularization * hidden_weights.array().pow(2).sum() +
			regularization * reconstruction_weights.array().pow(2).sum();

		auto hidden_delta = ((reconstruction_weights.transpose() * reconstruction_delta).cwiseProduct(hidden_activation.gradient(hidden_z))).eval();

		return{ loss,
			(hidden_delta * input.transpose() / input.cols()) + regularization * 2 * hidden_weights,
			hidden_delta.rowwise().sum() / input.cols(),
			(reconstruction_delta * hidden_a.transpose() / input.cols()) + regularization * 2 * reconstruction_weights,
			reconstruction_delta.rowwise().sum() / input.cols() };
	}

	template <class HiddenActivation, class ReconstructionActivation>
	auto sparse_loss(const HiddenActivation& hidden_activation, const ReconstructionActivation& reconstruction_activation,
	MatrixXdRef hidden_weights, VectorXdRef hidden_intercepts,
	MatrixXdRef reconstruction_weights, VectorXdRef reconstruction_intercepts,
	double regularization, double sparsity, double sparsity_weight,
	MatrixXdRef input, MatrixXdRef target) {
		auto hidden_z = ((hidden_weights * input).colwise() + hidden_intercepts).eval();
		auto hidden_a = (hidden_activation.compute(hidden_z)).eval();
		auto reconstruction_z = ((reconstruction_weights * hidden_a).colwise() + reconstruction_intercepts).eval();

		auto rho_hat = ((hidden_a.rowwise().sum() / input.cols()).unaryExpr([](double x) { return abs(x - 1.0) < numeric_limits<double>::epsilon() ? (x + numeric_limits<double>::epsilon()) : x; })).eval();
		auto sparsity_penalty = ((sparsity * (sparsity / rho_hat.array()).log()) + ((1 - sparsity) * ((1 - sparsity) / (1 - rho_hat.array())).log())).sum();

		return (((reconstruction_activation.compute(reconstruction_z)) - target).array().pow(2).sum() / (2 * input.cols())) +
			regularization * hidden_weights.array().pow(2).sum() +
			regularization * reconstruction_weights.array().pow(2).sum() +
			sparsity_weight * sparsity_penalty;
	}

	template <class HiddenActivation, class ReconstructionActivation>
	tuple<MatrixXd, VectorXd, MatrixXd, VectorXd>
	sparse_gradient(const HiddenActivation& hidden_activation, const ReconstructionActivation& reconstruction_activation,
	MatrixXdRef hidden_weights, VectorXdRef hidden_intercepts,
	MatrixXdRef reconstruction_weights, VectorXdRef reconstruction_intercepts,
	double regularization, double sparsity, double sparsity_weight,
	MatrixXdRef input, MatrixXdRef target) {
		auto hidden_z = ((hidden_weights * input).colwise() + hidden_intercepts).eval();
		auto hidden_a = (hidden_activation.compute(hidden_z)).eval();
		auto reconstruction_z = ((reconstruction_weights * hidden_a).colwise() + reconstruction_intercepts).eval();
		auto recontstruction_error = ((reconstruction_activation.compute(reconstruction_z) - target)).eval();
		auto reconstruction_delta = (recontstruction_error.cwiseProduct(reconstruction_activation.gradient(reconstruction_z))).eval();

		auto rho_hat = ((hidden_a.rowwise().sum() / input.cols()).unaryExpr([](double x) { return abs(x - 1.0) < numeric_limits<double>::epsilon() ? (x + numeric_limits<double>::epsilon()) : x; })).eval();
		auto sparsity_delta = ((-sparsity / rho_hat.array()) + ((1 - sparsity) / (1 - rho_hat.array()))).matrix().eval();
		auto hidden_delta = (((reconstruction_weights.transpose() * reconstruction_delta).colwise() +
			(sparsity_weight * sparsity_delta)).cwiseProduct(hidden_activation.gradient(hidden_z))).eval();

		return{ (hidden_delta * input.transpose() / input.cols()) + regularization * 2 * hidden_weights,
			hidden_delta.rowwise().sum() / input.cols(),
			(reconstruction_delta * hidden_a.transpose() / input.cols()) + regularization * 2 * reconstruction_weights,
			reconstruction_delta.rowwise().sum() / input.cols() };
	}

	template <class HiddenActivation, class ReconstructionActivation>
	tuple<double, MatrixXd, VectorXd, MatrixXd, VectorXd>
	sparse_loss_and_gradient(const HiddenActivation& hidden_activation, const ReconstructionActivation& reconstruction_activation,
	MatrixXdRef hidden_weights, VectorXdRef hidden_intercepts,
	MatrixXdRef reconstruction_weights, VectorXdRef reconstruction_intercepts,
	double regularization, double sparsity, double sparsity_weight,
	MatrixXdRef input, MatrixXdRef target) {
		auto hidden_z = ((hidden_weights * input).colwise() + hidden_intercepts).eval();
		auto hidden_a = (hidden_activation.compute(hidden_z)).eval();
		auto reconstruction_z = ((reconstruction_weights * hidden_a).colwise() + reconstruction_intercepts).eval();
		auto recontstruction_error = ((reconstruction_activation.compute(reconstruction_z) - target)).eval();
		auto reconstruction_delta = (recontstruction_error.cwiseProduct(reconstruction_activation.gradient(reconstruction_z))).eval();

		auto rho_hat = ((hidden_a.rowwise().sum() / input.cols()).unaryExpr([](double x) { return abs(x - 1.0) < numeric_limits<double>::epsilon() ? (x + numeric_limits<double>::epsilon()) : x; })).eval();
		auto sparsity_delta = ((-sparsity / rho_hat.array()) + ((1 - sparsity) / (1 - rho_hat.array()))).eval().matrix();
		auto hidden_delta = (((reconstruction_weights.transpose() * reconstruction_delta).colwise() +
			(sparsity_weight * sparsity_delta)).cwiseProduct(hidden_activation.gradient(hidden_z))).eval();

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