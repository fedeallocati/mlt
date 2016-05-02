#ifndef MLT_MODELS_TRANSFORMERS_SPARSE_TIED_AUTOENCODER_HPP
#define MLT_MODELS_TRANSFORMERS_SPARSE_TIED_AUTOENCODER_HPP

#include <Eigen/Core>

#include "../base_model.hpp"
#include "transformer_mixin.hpp"
#include "../implementations/autoencoder.hpp"

namespace mlt {
namespace models {
namespace transformers {
	template <class HiddenActivation, class ReconstructionActivation, class Optimizer>
	class SparseTiedAutoencoder : public BaseModel, public TransformerMixin<SparseTiedAutoencoder<HiddenActivation, ReconstructionActivation, Optimizer>> {
	public:
		template <typename H, typename R, typename O,
			class = std::enable_if<std::is_convertible<std::decay_t<H>, HiddenActivation>::value
			&& std::is_convertible<std::decay_t<R>, ReconstructionActivation>::value
			&& std::is_convertible<std::decay_t<O>, Optimizer>::value>>
		explicit SparseTiedAutoencoder(int hidden_units, H&& hidden_activation, R&& reconstruction_activation, O&& optimizer, double regularization,
		double sparsity, double sparsity_weight) : _hidden_units(hidden_units), _hidden_activation(std::forward<H>(hidden_activation)),
			_reconstruction_activation(std::forward<R>(reconstruction_activation)), _optimizer(std::forward<O>(optimizer)), _regularization(regularization),
			_sparsity(sparsity), _sparsity_weight(sparsity_weight) {}

		Eigen::MatrixXd transform(const Eigen::MatrixXd& input) const {
			assert(this->_fitted);
			return _hidden_activation.compute((this->_weights * input).colwise() + this->_hidden_intercepts);
		}

		SparseTiedAutoencoder& fit(const Eigen::Ref<const Eigen::MatrixXd>& input, bool cold_start = true) {
			Eigen::VectorXd init(_hidden_units * input.rows() + _hidden_units + input.rows());

			if (this->_fitted && !cold_start) {
				init.block(0, 0, this->_weights.size(), 1) = utils::eigen::ravel(this->_weights);
				init.block(this->_weights.size(), 0, this->_hidden_intercepts.size(), 1) = this->_hidden_intercepts;

				init.block(this->_weights.size() + this->_hidden_intercepts.size(),
					0, this->_reconstruction_intercepts.size(), 1) = this->_reconstruction_intercepts;
			}
			else {
				init = (init.setRandom() * 4 / std::sqrt(6.0 / (_hidden_units + input.rows())));
			}

			Eigen::VectorXd coeffs = _optimizer.run(*this, input, input, init, cold_start);

			this->_weights = utils::eigen::unravel(coeffs.block(0, 0, _hidden_units * input.rows(), 1), _hidden_units, input.rows());
			this->_hidden_intercepts = coeffs.block(_hidden_units * input.rows(), 0, _hidden_units, 1);
			this->_reconstruction_intercepts = coeffs.block(_hidden_units * input.rows() + _hidden_units, 0, input.rows(), 1);

			this->_fitted = true;
			this->_input_size = input.rows();
			this->_output_size = _hidden_units;

			return *this;
		}

		using TransformerMixin<SparseTiedAutoencoder<HiddenActivation, ReconstructionActivation, Optimizer>>::fit;

		auto loss(const Eigen::Ref<const Eigen::VectorXd>& coeffs, const Eigen::Ref<const Eigen::MatrixXd>& input, const Eigen::Ref<const Eigen::MatrixXd>& target) const {
			auto weights = utils::eigen::unravel(coeffs.block(0, 0, _hidden_units * input.rows(), 1), _hidden_units, input.rows());
			auto hidden_intercepts = coeffs.block(_hidden_units * input.rows(), 0, _hidden_units, 1);
			auto reconstruction_intercepts = coeffs.block(_hidden_units * input.rows() + _hidden_units, 0, input.rows(), 1);

			return implementations::autoencoder::sparse_loss(_hidden_activation, _reconstruction_activation, weights, hidden_intercepts,
				weights.transpose(), reconstruction_intercepts, this->_regularization, this->_sparsity, this->_sparsity_weight, input, target);
		}

		auto gradient(const Eigen::Ref<const Eigen::VectorXd>& coeffs, const Eigen::Ref<const Eigen::MatrixXd>& input, const Eigen::Ref<const Eigen::MatrixXd>& target) const {
			auto weights = utils::eigen::unravel(coeffs.block(0, 0, _hidden_units * input.rows(), 1), _hidden_units, input.rows());
			auto hidden_intercepts = coeffs.block(_hidden_units * input.rows(), 0, _hidden_units, 1);
			auto reconstruction_intercepts = coeffs.block(_hidden_units * input.rows() + _hidden_units, 0, input.rows(), 1);

			Eigen::MatrixXd weights_grad, weights_transp_grad;
			Eigen::VectorXd hid_inter_grad, rec_inter_grad;
			std::tie(weights_grad, hid_inter_grad, weights_transp_grad, rec_inter_grad) = implementations::autoencoder::sparse_gradient(_hidden_activation,
				_reconstruction_activation, weights, hidden_intercepts, weights.transpose(), reconstruction_intercepts, this->_regularization, 
				this->_sparsity, this->_sparsity_weight, input, target);

			Eigen::VectorXd gradient(coeffs.rows());

			gradient.block(0, 0, weights_grad.size(), 1) = utils::eigen::ravel(weights_grad + weights_transp_grad.transpose());
			gradient.block(weights_grad.size(), 0, hid_inter_grad.size(), 1) = hid_inter_grad;

			gradient.block(weights_grad.size() + hid_inter_grad.size(),
				0, rec_inter_grad.size(), 1) = rec_inter_grad;

			return gradient;
		}

		std::tuple<double, Eigen::VectorXd> loss_and_gradient(const Eigen::Ref<const Eigen::VectorXd>& coeffs, const Eigen::Ref<const Eigen::MatrixXd>& input, const Eigen::Ref<const Eigen::MatrixXd>& target) const {
			auto weights = utils::eigen::unravel(coeffs.block(0, 0, _hidden_units * input.rows(), 1), _hidden_units, input.rows());
			auto hidden_intercepts = coeffs.block(_hidden_units * input.rows(), 0, _hidden_units, 1);
			auto reconstruction_intercepts = coeffs.block(_hidden_units * input.rows() + _hidden_units, 0, input.rows(), 1);

			double loss;
			Eigen::MatrixXd weights_grad, weights_transp_grad;
			Eigen::VectorXd hid_inter_grad, rec_inter_grad;
			std::tie(loss, weights_grad, hid_inter_grad, weights_transp_grad, rec_inter_grad) = implementations::autoencoder::sparse_loss_and_gradient(
				_hidden_activation, _reconstruction_activation, weights, hidden_intercepts, weights.transpose(), reconstruction_intercepts,
				this->_regularization, this->_sparsity, this->_sparsity_weight, input, target);

			Eigen::VectorXd gradient(coeffs.rows());

			gradient.block(0, 0, weights_grad.size(), 1) = utils::eigen::ravel(weights_grad + weights_transp_grad.transpose());
			gradient.block(weights_grad.size(), 0, hid_inter_grad.size(), 1) = hid_inter_grad;

			gradient.block(weights_grad.size() + hid_inter_grad.size(),
				0, rec_inter_grad.size(), 1) = rec_inter_grad;

			return { loss, gradient };
		}

	protected:
		int _hidden_units;
		HiddenActivation _hidden_activation;
		ReconstructionActivation _reconstruction_activation;
		Optimizer _optimizer;
		double _regularization;
		double _sparsity;
		double _sparsity_weight;

		Eigen::MatrixXd _weights;
		Eigen::VectorXd _hidden_intercepts;
		Eigen::VectorXd _reconstruction_intercepts;
	};

	template <class HiddenActivation, class ReconstructionActivation, class Optimizer>
	auto create_sparse_tied_autoencoder(int hidden_units, HiddenActivation&& hidden_activation,
	ReconstructionActivation&& reconstruction_activation, Optimizer&& optimizer,
	double regularization, double sparsity, double sparsity_weight) {
		return SparseTiedAutoencoder<HiddenActivation, ReconstructionActivation, Optimizer>(
			hidden_units,
			std::forward<HiddenActivation>(hidden_activation), 
			std::forward<ReconstructionActivation>(reconstruction_activation),
			std::forward<Optimizer>(optimizer),
			regularization, sparsity, sparsity_weight);
	}
}
}
}
#endif