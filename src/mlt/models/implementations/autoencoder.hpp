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
	const Eigen::Ref<const Eigen::MatrixXd>& input, const Eigen::Ref<const Eigen::MatrixXd>& target) {
		return (((reconstruction_activation.compute((reconstruction_weights * (hidden_activation.compute((hidden_weights * input).colwise() + hidden_intercepts))).colwise() + reconstruction_intercepts)) - target).array().pow(2).sum() / (2 * input.cols()));
	}

	template <class HiddenActivation, class ReconstructionActivation>
	std::tuple<Eigen::MatrixXd, Eigen::VectorXd, Eigen::MatrixXd, Eigen::VectorXd> 
	gradient(const HiddenActivation& hidden_activation, const ReconstructionActivation& reconstruction_activation,
	const Eigen::Ref<const Eigen::MatrixXd>& hidden_weights, const Eigen::Ref<const Eigen::VectorXd>& hidden_intercepts,
	const Eigen::Ref<const Eigen::MatrixXd>& reconstruction_weights, const Eigen::Ref<const Eigen::VectorXd>& reconstruction_intercepts,
	const Eigen::Ref<const Eigen::MatrixXd>& input, const Eigen::Ref<const Eigen::MatrixXd>& target) {
		Eigen::MatrixXd hidden_z = (hidden_weights * input).colwise() + hidden_intercepts;
		Eigen::MatrixXd hidden_activation_val = hidden_activation.compute(hidden_z);
		Eigen::MatrixXd reconstruction_z = (reconstruction_weights * hidden_activation_val).colwise() + reconstruction_intercepts;

		Eigen::MatrixXd recontstruction_error = (reconstruction_activation.compute(reconstruction_z) - target) / input.cols();

		Eigen::MatrixXd reconstruction_delta = recontstruction_error.cwiseProduct(reconstruction_activation.gradient(reconstruction_z));
		Eigen::MatrixXd hidden_delta = (hidden_weights * reconstruction_delta).cwiseProduct(hidden_activation.gradient(hidden_z));

		return { hidden_delta * input.transpose(), hidden_delta.rowwise().sum(), reconstruction_delta * hidden_activation_val.transpose(), reconstruction_delta.rowwise().sum() };
	}

	template <class HiddenActivation, class ReconstructionActivation>
	std::tuple<double, Eigen::MatrixXd, Eigen::VectorXd, Eigen::MatrixXd, Eigen::VectorXd>
	loss_and_gradient(const HiddenActivation& hidden_activation, const ReconstructionActivation& reconstruction_activation,
	const Eigen::Ref<const Eigen::MatrixXd>& hidden_weights, const Eigen::Ref<const Eigen::VectorXd>& hidden_intercepts,
	const Eigen::Ref<const Eigen::MatrixXd>& reconstruction_weights, const Eigen::Ref<const Eigen::VectorXd>& reconstruction_intercepts,
	const Eigen::Ref<const Eigen::MatrixXd>& input, const Eigen::Ref<const Eigen::MatrixXd>& target) {
		Eigen::MatrixXd hidden_z = (hidden_weights * input).colwise() + hidden_intercepts;
		Eigen::MatrixXd hidden_activation_val = hidden_activation.compute(hidden_z);
		Eigen::MatrixXd reconstruction_z = (reconstruction_weights * hidden_activation_val).colwise() + reconstruction_intercepts;
		Eigen::MatrixXd reconstruction_activation_val = reconstruction_activation.compute(reconstruction_z);

		Eigen::MatrixXd recontstruction_error = (reconstruction_activation_val - target) / input.cols();

		Eigen::MatrixXd reconstruction_delta = recontstruction_error.cwiseProduct(reconstruction_activation.gradient(reconstruction_z));
		Eigen::MatrixXd hidden_delta = (hidden_weights * reconstruction_delta).cwiseProduct(hidden_activation.gradient(hidden_z));

		double loss = ((reconstruction_activation_val - target).array().pow(2).sum() / (2 * input.cols())) + _regularization * (coeffs.block(0, 0, coeffs.rows() - 1, coeffs.cols() - 1).array().pow(2).sum());

		return { loss, hidden_delta * input.transpose(), hidden_delta.rowwise().sum(), reconstruction_delta * hidden_activation_val.transpose(), reconstruction_delta.rowwise().sum() };
	}

	/*template <class HiddenActivation, class ReconstructionActivation, class Optimizer>
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
	};*/
}
}
}
}
#endif