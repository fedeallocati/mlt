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
			Eigen::MatrixXd init(_hidden_units + 1, input.rows() + 1);
			if (this->_fitted && !cold_start) {
				init.block(0, 0, init.rows() - 1, init.cols() - 1) = this->_weights;
				init.rightCols<1>().head(init.rows() - 1) = this->_hidden_intercepts;
				init.bottomRows<1>().head(init.cols() - 1) = this->_reconstruction_intercepts.transpose();
			}
			else {
				init = (init.setRandom() * 4 / std::sqrt(6.0 / (_hidden_units + input.rows())));
			}

			Eigen::MatrixXd coeffs = _optimizer.run(*this, input, input, init, cold_start);

			this->_weights = coeffs.block(0, 0, coeffs.rows() - 1, coeffs.cols() - 1);
			this->_hidden_intercepts = coeffs.rightCols<1>().head(coeffs.rows() - 1);
			this->_reconstruction_intercepts = coeffs.bottomRows<1>().head(coeffs.cols() - 1).transpose();
			this->_fitted = true;
			this->_input_size = input.rows();
			this->_output_size = _hidden_units;

			return *this;
		}

		using TransformerMixin<SparseTiedAutoencoder<HiddenActivation, ReconstructionActivation, Optimizer>>::fit;

		auto loss(const Eigen::Ref<const Eigen::MatrixXd>& coeffs, const Eigen::Ref<const Eigen::MatrixXd>& input, const Eigen::Ref<const Eigen::MatrixXd>& target) const {
			auto weights = coeffs.block(0, 0, coeffs.rows() - 1, coeffs.cols() - 1);
			auto hidden_intercepts = coeffs.rightCols<1>().head(coeffs.rows() - 1);
			auto reconstruction_intercepts = coeffs.bottomRows<1>().head(coeffs.cols() - 1).transpose();

			return implementations::autoencoder::sparse_loss(_hidden_activation, _reconstruction_activation, weights, hidden_intercepts,
				weights.transpose(), reconstruction_intercepts, this->_regularization, this->_sparsity, this->_sparsity_weight, input, target);
		}

		auto gradient(const Eigen::Ref<const Eigen::MatrixXd>& coeffs, const Eigen::Ref<const Eigen::MatrixXd>& input, const Eigen::Ref<const Eigen::MatrixXd>& target) const {
			auto weights = coeffs.block(0, 0, coeffs.rows() - 1, coeffs.cols() - 1);
			auto hidden_intercepts = coeffs.rightCols<1>().head(coeffs.rows() - 1);
			auto reconstruction_intercepts = coeffs.bottomRows<1>().head(coeffs.cols() - 1).transpose();

			Eigen::MatrixXd coeff_grad, hid_inter_grad, coeff_transp_grad, rec_inter_grad;
			std::tie(coeff_grad, hid_inter_grad, coeff_transp_grad, rec_inter_grad) = implementations::autoencoder::sparse_gradient(_hidden_activation,
				_reconstruction_activation, weights, hidden_intercepts, weights.transpose(), reconstruction_intercepts, this->_regularization, 
				this->_sparsity, this->_sparsity_weight, input, target);

			Eigen::MatrixXd gradient = Eigen::MatrixXd::Zero(coeffs.rows(), coeffs.cols());
			gradient.block(0, 0, coeffs.rows() - 1, coeffs.cols() - 1) = coeff_grad;
			gradient.rightCols<1>().head(coeffs.rows() - 1) = hid_inter_grad;
			gradient.block(0, 0, coeffs.rows() - 1, coeffs.cols() - 1) += coeff_transp_grad.transpose();
			gradient.bottomRows<1>().head(coeffs.cols() - 1) = rec_inter_grad.transpose();

			return gradient;
		}

		std::tuple<double, Eigen::MatrixXd> loss_and_gradient(const Eigen::Ref<const Eigen::MatrixXd>& coeffs, const Eigen::Ref<const Eigen::MatrixXd>& input, const Eigen::Ref<const Eigen::MatrixXd>& target) const {
			auto weights = coeffs.block(0, 0, coeffs.rows() - 1, coeffs.cols() - 1);
			auto hidden_intercepts = coeffs.rightCols<1>().head(coeffs.rows() - 1);
			auto reconstruction_intercepts = coeffs.bottomRows<1>().head(coeffs.cols() - 1).transpose();

			double loss;
			Eigen::MatrixXd coeff_grad, hid_inter_grad, coeff_transp_grad, rec_inter_grad;
			std::tie(loss, coeff_grad, hid_inter_grad, coeff_transp_grad, rec_inter_grad) = implementations::autoencoder::sparse_loss_and_gradient(
				_hidden_activation, _reconstruction_activation, weights, hidden_intercepts, weights.transpose(), reconstruction_intercepts,
				this->_regularization, this->_sparsity, this->_sparsity_weight, input, target);

			Eigen::MatrixXd gradient = Eigen::MatrixXd::Zero(coeffs.rows(), coeffs.cols());
			gradient.block(0, 0, coeffs.rows() - 1, coeffs.cols() - 1) = coeff_grad;
			gradient.rightCols<1>().head(coeffs.rows() - 1) = hid_inter_grad;
			gradient.block(0, 0, coeffs.rows() - 1, coeffs.cols() - 1) += coeff_transp_grad.transpose();
			gradient.bottomRows<1>().head(coeffs.cols() - 1) = rec_inter_grad.transpose();

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