#ifndef MLT_MODELS_TRANSFORMERS_AUTOENCODER_HPP
#define MLT_MODELS_TRANSFORMERS_AUTOENCODER_HPP

#include <cmath>
#include <tuple>
#include <type_traits>

#include <Eigen/Core>

#include "transformer.hpp"
#include "../implementations/autoencoder.hpp"
#include "../../utils/eigen.hpp"

namespace mlt {
namespace models {
namespace transformers {
	using namespace utils::eigen;

	template <class HiddenActivation, class ReconstructionActivation, class Optimizer>
	class Autoencoder : public Transformer<Autoencoder<HiddenActivation, ReconstructionActivation, Optimizer>> {
	public:
		template <typename H, typename R, typename O,
			class = enable_if<is_convertible<decay_t<H>, HiddenActivation>::value
			&& is_convertible<decay_t<R>, ReconstructionActivation>::value
			&& is_convertible<decay_t<O>, Optimizer>::value>>
		explicit Autoencoder(int hidden_units, H&& hidden_activation, R&& reconstruction_activation, O&& optimizer, double regularization) :
			_hidden_units(hidden_units), _hidden_activation(forward<H>(hidden_activation)), _reconstruction_activation(forward<R>(reconstruction_activation)),
			_optimizer(forward<O>(optimizer)), _regularization(regularization) {}

		Result transform(Features input) const {
			assert(_fitted);
			return _hidden_activation.compute((_hidden_weights * input).colwise() + _hidden_intercepts);
		}

		Self& fit(Features input, bool cold_start = true) {
			VectorXd init(_hidden_units * input.rows() + _hidden_units + input.rows() * _hidden_units + input.rows());

			if (_fitted && !cold_start) {
				init.block(0, 0, _hidden_weights.size(), 1) = ravel(_hidden_weights);
				init.block(_hidden_weights.size(), 0, _hidden_intercepts.size(), 1) = _hidden_intercepts;

				init.block(_hidden_weights.size() + _hidden_intercepts.size(), 0, _reconstruction_weights.size(), 1) =
					ravel(_reconstruction_weights);
				init.block(_hidden_weights.size() + _hidden_intercepts.size() + _reconstruction_weights.size(),
					0, _reconstruction_intercepts.size(), 1) = _reconstruction_intercepts;
			} else {
				init = (init.setRandom() * 4 / sqrt(6.0 / (_hidden_units + input.rows())));
			}

			VectorXd coeffs = _optimizer(*this, input, input, init, cold_start);

			_hidden_weights = unravel(coeffs.block(0, 0, _hidden_units * input.rows(), 1), _hidden_units, input.rows());
			_hidden_intercepts = coeffs.block(_hidden_units * input.rows(), 0, _hidden_units, 1);
			_reconstruction_weights = unravel(coeffs.block(_hidden_units * input.rows() + _hidden_units, 0, _hidden_units * input.rows(), 1), input.rows(), _hidden_units);
			_reconstruction_intercepts = coeffs.block(_hidden_units * input.rows() + _hidden_units + input.rows() * _hidden_units, 0, input.rows(), 1);

			_fitted = true;

			return _self();
		}

		using Transformer<Self>::fit;

		auto loss(VectorXdRef coeffs, Features input, Features target) const {
			auto hidden_weights = unravel(coeffs.block(0, 0, _hidden_units * input.rows(), 1), _hidden_units, input.rows());
			auto hidden_intercepts = coeffs.block(_hidden_units * input.rows(), 0, _hidden_units, 1);
			auto reconstruction_weights = unravel(coeffs.block(_hidden_units * input.rows() + _hidden_units, 0, input.rows() * _hidden_units, 1), input.rows(), _hidden_units);
			auto reconstruction_intercepts = coeffs.block(_hidden_units * input.rows() + _hidden_units + input.rows() * _hidden_units, 0, input.rows(), 1);

			return implementations::autoencoder::loss(_hidden_activation, _reconstruction_activation, hidden_weights, hidden_intercepts,
				reconstruction_weights, reconstruction_intercepts, _regularization, input, target);
		}

		auto gradient(VectorXdRef coeffs, Features input, Features target) const {
			auto hidden_weights = unravel(coeffs.block(0, 0, _hidden_units * input.rows(), 1), _hidden_units, input.rows());
			auto hidden_intercepts = coeffs.block(_hidden_units * input.rows(), 0, _hidden_units, 1);
			auto reconstruction_weights = unravel(coeffs.block(_hidden_units * input.rows() + _hidden_units, 0, input.rows() * _hidden_units, 1), input.rows(), _hidden_units);
			auto reconstruction_intercepts = coeffs.block(_hidden_units * input.rows() + _hidden_units + input.rows() * _hidden_units, 0, input.rows(), 1);

			MatrixXd hid_weights_grad, rec_weights_grad;
			VectorXd hid_inter_grad, rec_inter_grad;

			tie(hid_weights_grad, hid_inter_grad, rec_weights_grad, rec_inter_grad) = implementations::autoencoder::gradient(_hidden_activation,
				_reconstruction_activation, hidden_weights, hidden_intercepts, reconstruction_weights, reconstruction_intercepts, _regularization,
				input, target);

			VectorXd g(coeffs.rows());

			g.block(0, 0, hid_weights_grad.size(), 1) = ravel(hid_weights_grad);
			g.block(hid_weights_grad.size(), 0, hid_inter_grad.size(), 1) = hid_inter_grad;

			g.block(hid_weights_grad.size() + hid_inter_grad.size(), 0, rec_weights_grad.size(), 1) =
					ravel(rec_weights_grad);
			g.block(hid_weights_grad.size() + hid_inter_grad.size() + rec_weights_grad.size(),
					0, rec_inter_grad.size(), 1) = rec_inter_grad;

			return g;
		}

		auto loss_and_gradient(VectorXdRef coeffs, Features input, Features target) const {
			auto hidden_weights = unravel(coeffs.block(0, 0, _hidden_units * input.rows(), 1), _hidden_units, input.rows());
			auto hidden_intercepts = coeffs.block(_hidden_units * input.rows(), 0, _hidden_units, 1);
			auto reconstruction_weights = unravel(coeffs.block(_hidden_units * input.rows() + _hidden_units, 0, input.rows() * _hidden_units, 1), input.rows(), _hidden_units);
			auto reconstruction_intercepts = coeffs.block(_hidden_units * input.rows() + _hidden_units + input.rows() * _hidden_units, 0, input.rows(), 1);

			double l;
			MatrixXd hid_weights_grad, rec_weights_grad;
			VectorXd hid_inter_grad, rec_inter_grad;

			tie(l, hid_weights_grad, hid_inter_grad, rec_weights_grad, rec_inter_grad) = implementations::autoencoder::loss_and_gradient(_hidden_activation,
				_reconstruction_activation, hidden_weights, hidden_intercepts, reconstruction_weights, reconstruction_intercepts, _regularization,
				input, target);

			VectorXd g(coeffs.rows());

			g.block(0, 0, hid_weights_grad.size(), 1) = ravel(hid_weights_grad);
			g.block(hid_weights_grad.size(), 0, hid_inter_grad.size(), 1) = hid_inter_grad;

			g.block(hid_weights_grad.size() + hid_inter_grad.size(), 0, rec_weights_grad.size(), 1) =
				ravel(rec_weights_grad);
			g.block(hid_weights_grad.size() + hid_inter_grad.size() + rec_weights_grad.size(),
				0, rec_inter_grad.size(), 1) = rec_inter_grad;

			return make_tuple(l, g);
		}

	protected:
		int _hidden_units;
		HiddenActivation _hidden_activation;
		ReconstructionActivation _reconstruction_activation;
		Optimizer _optimizer;
		double _regularization;

		MatrixXd _hidden_weights;
		VectorXd _hidden_intercepts;
		MatrixXd _reconstruction_weights;
		VectorXd _reconstruction_intercepts;
	};

	template <class HiddenActivation, class ReconstructionActivation, class Optimizer>
	auto create_autoencoder(int hidden_units, HiddenActivation&& hidden_activation, 
	ReconstructionActivation&& reconstruction_activation, Optimizer&& optimizer,
	double regularization) {
		return Autoencoder<HiddenActivation, ReconstructionActivation, Optimizer>(
			hidden_units,
			forward<HiddenActivation>(hidden_activation), 
			forward<ReconstructionActivation>(reconstruction_activation),
			forward<Optimizer>(optimizer),
			regularization);
	}
}
}
}
#endif