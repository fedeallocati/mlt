#ifndef MLT_MODELS_TRANSFORMERS_AUTOENCODER_HPP
#define MLT_MODELS_TRANSFORMERS_AUTOENCODER_HPP

#include <Eigen/Core>

#include "../base_model.hpp"
#include "transformer_mixin.hpp"

namespace mlt {
namespace models {
namespace transformers {
	template <class HiddenActivation, class ReconstructionActivation, class Optimizer>
	class Autoencoder : public BaseModel, public TransformerMixin<Autoencoder<HiddenActivation, ReconstructionActivation, Optimizer>> {
	public:
		template <typename H, typename R, typename O,
			class = std::enable_if<std::is_same<std::decay_t<H>, HiddenActivation>::value 
			&& std::is_convertible<std::decay_t<R>, ReconstructionActivation>::value
			&& std::is_convertible<std::decay_t<O>, Optimizer>::value>>
		explicit Autoencoder(int hidden_units, H&& hidden_activation, R&& reconstruction_activation, O&& optimizer, double regularization) :
			_hidden_units(hidden_units), _hidden_activation(std::forward<H>(hidden_activation)), _reconstruction_activation(std::forward<R>(reconstruction_activation)),
			_optimizer(std::forward<O>(optimizer)), _regularization(regularization) {}

		Eigen::MatrixXd transform(const Eigen::MatrixXd& input) const {
			assert(this->_fitted);
			return this->_compute_hidden_activation(input, this->_coefficients, this->_intercepts);
		}

		Autoencoder& fit(const Eigen::Ref<const Eigen::MatrixXd>& input, bool cold_start = true) {
			Eigen::MatrixXd init = this->_fitted && !cold_start ? _coefficients :
				(Eigen::MatrixXd::Random(_hidden_units + 1, input.rows() + 1) * 4 / std::sqrt(6.0 / (_hidden_units + input.rows())));

			Eigen::MatrixXd coeffs = _optimizer.run(*this, input, input, init, cold_start);

			this->_coefficients = coeffs.leftCols(coeffs.cols() - 1);
			this->_intercepts = coeffs.rightCols<2>().leftCols<1>();
			this->_intercepts_prime = coeffs.rightCols<1>();
			this->_fitted = true;
			this->_input_size = input.rows();
			this->_output_size = _hidden_units;

			return *this;
		}

		using TransformerMixin<Autoencoder<HiddenActivation, ReconstructionActivation, Optimizer>>::fit;

		double loss(const Eigen::Ref<const Eigen::MatrixXd>& coeffs, const Eigen::Ref<const Eigen::MatrixXd>& input, const Eigen::Ref<const Eigen::MatrixXd>& target) const {
			auto coefficients = coeffs.block(0, 0, coeffs.rows() - 1, coeffs.cols() - 1);
			auto hidden_intercepts = coeffs.rightCols<1>().head(coeffs.rows() - 1);
			auto output_intercepts = coeffs.bottomRows<1>().head(coeffs.cols() - 1).transpose();

			Eigen::MatrixXd hidden_activation = _compute_hidden_activation(input, coefficients, hidden_intercepts);
			Eigen::MatrixXd reconstruction_activation = _compute_reconstruction_activation(hidden_activation, coefficients.transpose(), output_intercepts);
			return ((reconstruction_activation - target).array().pow(2).sum() / (2 * input.cols())) + _regularization * coefficients.array().pow(2).sum();
		}

		Eigen::MatrixXd gradient(const Eigen::Ref<const Eigen::MatrixXd>& coeffs, const Eigen::Ref<const Eigen::MatrixXd>& input, const Eigen::Ref<const Eigen::MatrixXd>& target) const {	
			auto coefficients = coeffs.block(0, 0, coeffs.rows() - 1, coeffs.cols() - 1);
			auto hidden_intercepts = coeffs.rightCols<1>().head(coeffs.rows() - 1);
			auto output_intercepts = coeffs.bottomRows<1>().head(coeffs.cols() - 1).transpose();
			
			Eigen::MatrixXd hidden_z = (coefficients * input).colwise() + hidden_intercepts;
			Eigen::MatrixXd hidden_activation = _hidden_activation.compute(hidden_z);
			Eigen::MatrixXd reconstruction_z = (coefficients.transpose() * hidden_activation).colwise() + output_intercepts;
			Eigen::MatrixXd reconstruction_activation = _reconstruction_activation.compute(reconstruction_z);

			Eigen::MatrixXd recontstruction_error = (reconstruction_activation - target) / input.cols();

			Eigen::MatrixXd reconstruction_delta = recontstruction_error.cwiseProduct(_reconstruction_activation.gradient(reconstruction_z));
			Eigen::MatrixXd hidden_delta = (coefficients * reconstruction_delta).cwiseProduct(_hidden_activation.gradient(hidden_z));

			Eigen::MatrixXd gradient = Eigen::MatrixXd::Zero(coeffs.rows(), coeffs.cols());
			// Gradients of coefficients.transpose() and output_intercepts
			gradient.block(0, 0, coeffs.rows() - 1, coeffs.cols() - 1) = hidden_activation * reconstruction_delta.transpose();
			gradient.bottomRows<1>().head(coeffs.cols() - 1) = reconstruction_delta.rowwise().sum().transpose();
			// Gradients of coefficients and hidden_intercepts
			gradient.block(0, 0, coeffs.rows() - 1, coeffs.cols() - 1) += hidden_delta * input.transpose();
			gradient.rightCols<1>().head(coeffs.rows() - 1) = hidden_delta.rowwise().sum();

			gradient.block(0, 0, coeffs.rows() - 1, coeffs.cols() - 1) += _regularization * 2 * coefficients;

			return gradient;
		}

		std::tuple<double, Eigen::MatrixXd> loss_and_gradient(const Eigen::Ref<const Eigen::MatrixXd>& coeffs, const Eigen::Ref<const Eigen::MatrixXd>& input, const Eigen::Ref<const Eigen::MatrixXd>& target) const {
			auto coefficients = coeffs.block(0, 0, coeffs.rows() - 1, coeffs.cols() - 1);
			auto hidden_intercepts = coeffs.rightCols<1>().head(coeffs.rows() - 1);
			auto output_intercepts = coeffs.bottomRows<1>().head(coeffs.cols() - 1).transpose();

			Eigen::MatrixXd hidden_z = (coefficients * input).colwise() + hidden_intercepts;
			Eigen::MatrixXd hidden_activation = _hidden_activation.compute(hidden_z);
			Eigen::MatrixXd reconstruction_z = (coefficients.transpose() * hidden_activation).colwise() + output_intercepts;
			Eigen::MatrixXd reconstruction_activation = _reconstruction_activation.compute(reconstruction_z);

			Eigen::MatrixXd recontstruction_error = (reconstruction_activation - target) / input.cols();

			Eigen::MatrixXd reconstruction_delta = recontstruction_error.cwiseProduct(_reconstruction_activation.gradient(reconstruction_z));
			Eigen::MatrixXd hidden_delta = (coefficients * reconstruction_delta).cwiseProduct(_hidden_activation.gradient(hidden_z));

			double loss = ((reconstruction_activation - target).array().pow(2).sum() / (2 * input.cols())) + _regularization * (coeffs.block(0, 0, coeffs.rows() - 1, coeffs.cols() - 1).array().pow(2).sum());
			Eigen::MatrixXd gradient = Eigen::MatrixXd::Zero(coeffs.rows(), coeffs.cols());
			// Gradients of coefficients.transpose() and output_intercepts
			gradient.block(0, 0, coeffs.rows() - 1, coeffs.cols() - 1) = hidden_activation * reconstruction_delta.transpose();
			gradient.bottomRows<1>().head(coeffs.cols() - 1) = reconstruction_delta.rowwise().sum().transpose();
			// Gradients of coefficients and hidden_intercepts
			gradient.block(0, 0, coeffs.rows() - 1, coeffs.cols() - 1) += hidden_delta * input.transpose();
			gradient.rightCols<1>().head(coeffs.rows() - 1) = hidden_delta.rowwise().sum();

			gradient.block(0, 0, coeffs.rows() - 1, coeffs.cols() - 1) += _regularization * 2 * coefficients;

			return { loss, gradient };
		}

	protected:
		Eigen::MatrixXd _compute_hidden_activation(const Eigen::Ref<const Eigen::MatrixXd>& input, const Eigen::Ref<const Eigen::MatrixXd>& coefficients, const Eigen::Ref<const Eigen::VectorXd>& intercepts) const {
			return _hidden_activation.compute((coefficients * input).colwise() + intercepts);
		}

		Eigen::MatrixXd _compute_reconstruction_activation(const Eigen::Ref<const Eigen::MatrixXd>& input, const Eigen::Ref<const Eigen::MatrixXd>& coefficients, const Eigen::Ref<const Eigen::VectorXd>& intercepts) const {
			return _reconstruction_activation.compute((coefficients * input).colwise() + intercepts);
		}

		int _hidden_units;
		HiddenActivation _hidden_activation;
		ReconstructionActivation _reconstruction_activation;
		Optimizer _optimizer;
		double _regularization;
		bool _fit_intercept;

		Eigen::MatrixXd _coefficients;
		Eigen::VectorXd _intercepts;
		Eigen::VectorXd _intercepts_prime;
	};
}
}
}
#endif