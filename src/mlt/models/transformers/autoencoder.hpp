#ifndef MLT_MODELS_TRANSFORMERS_AUTOENCODER_HPP
#define MLT_MODELS_TRANSFORMERS_AUTOENCODER_HPP

#include <Eigen/Core>

#include "../base_model.hpp"
#include "transformer_mixin.hpp"

namespace mlt {
namespace models {
namespace transformers {
	template <class Activation, class Optimizer>
	class Autoencoder : public BaseModel, public TransformerMixin<Autoencoder<Activation, Optimizer>> {
	public:
		explicit Autoencoder(int hidden_units, const Activation& activation, const Optimizer& optimizer, double regularization) :
			_hidden_units(hidden_units), _activation(activation), _optimizer(optimizer), _regularization(regularization) {}

		explicit Autoencoder(int hidden_units, Activation&& activation, const Optimizer& optimizer, double regularization) :
			_hidden_units(hidden_units), _activation(activation), _optimizer(optimizer), _regularization(regularization) {}

		explicit Autoencoder(int hidden_units, const Activation& activation, Optimizer&& optimizer, double regularization) :
			_hidden_units(hidden_units), _activation(activation), _optimizer(optimizer), _regularization(regularization) {}

		explicit Autoencoder(int hidden_units, Activation&& activation, Optimizer&& optimizer, double regularization) :
			_hidden_units(hidden_units), _activation(activation), _optimizer(optimizer), _regularization(regularization) {}

		Autoencoder& fit(const Eigen::Ref<const Eigen::MatrixXd>& input, bool cold_start = true) {
			Eigen::MatrixXd init = this->_fitted && !cold_start ? _coefficients :
				(Eigen::MatrixXd::Random(_hidden_units + 1, input.rows() + 1) * 4 / std::sqrt(6.0 / (_hidden_units + input.rows())));

			Eigen::MatrixXd coeffs = _optimizer.run(*this, input, input, init, cold_start);

			_coefficients = coeffs.leftCols(coeffs.cols() - 1);
			_intercepts = coeffs.rightCols<2>().leftCols<1>();
			_intercepts_prime = coeffs.rightCols<1>();
			_fitted = true;
			_input_size = input.rows();
			_output_size = _hidden_units;

			return *this;
		}

		Eigen::MatrixXd predict(const Eigen::MatrixXd& input) const {
			assert(_fitted);
			return _apply_linear_transformation(input);
		}

		double loss(const Eigen::Ref<const Eigen::MatrixXd>& coeffs, const Eigen::Ref<const Eigen::MatrixXd>& input, const Eigen::Ref<const Eigen::MatrixXd>& target) const {
			// input = (100, 5)
			// coeffs = (51, 101)
			// coefficients = (50, 100)
			// intercepts = (50, 1)
			// intercepts_prime = (100, 1)
			// activations = (50, 100) * (100, 5) = (50, 5)
			// reconstruction = ((50, 100)' * (50, 5)) + (100, 1) = (100, 5)

			Eigen::MatrixXd hidden_activation = _compute_activation(input, coeffs.block(0, 0, coeffs.rows() - 1, coeffs.cols() - 1), coeffs.rightCols<1>().head(coeffs.rows() - 1));
			Eigen::MatrixXd recontstruction_activation = _compute_activation(hidden_activation, coeffs.block(0, 0, coeffs.rows() - 1, coeffs.cols() - 1).transpose(), coeffs.bottomRows<1>().head(coeffs.cols() - 1).transpose());
			return ((recontstruction_activation - target).array().pow(2).sum() / (2 * input.cols())) + _regularization * (coeffs.block(0, 0, coeffs.rows() - 1, coeffs.cols() - 1).array().pow(2).sum());
		}

		Eigen::MatrixXd gradient(const Eigen::Ref<const Eigen::MatrixXd>& coeffs, const Eigen::Ref<const Eigen::MatrixXd>& input, const Eigen::Ref<const Eigen::MatrixXd>& target) const {
			// input = (100, 5)
			// coeffs = (51, 101)
			// coefficients = (50, 100)
			// intercepts = (50, 1)
			// intercepts_prime = (100, 1)
			// hidden_z = (50, 100) * (100, 5) = (50, 5)
			// hidden_activation = (50, 5)
			// recontstruction_z = ((50, 100)' * (50, 5)) + (100, 1) = ((100, 50) * (50, 5)) + (100, 1) = (100, 5) + (100, 1) = (100, 5)
			// recontstruction_activation = (100, 5)
			// recontstruction_error = (100, 5) - (100, 5) = (100, 5)
			// recontstruction_delta = ((50, 100) * (100, 5)) .* (50, 5) = (50, 5) .* (50, 5) = (50, 5)
			// hidden_delta = ((50, 100)' * (50, 5)) .* (100, 5) = ((100, 50) * (50, 5)) .* (100, 5) = (100, 5) .* (100, 5) = (100, 5)

			// gradient(0:50, 0:100) = (50, 5) * (100, 5)' = (50, 100)
			// gradient(51, 0:100) = (rowwiseSum(100, 5))' = (100, 1)' = (1, 100)
			// gradient(0:50, 0:100) += ((50, 5) * (100, 5)') = (50, 100) 
			// gradient(0:50, 101) = (rowwiseSum(50, 5)) = (50, 1)

			Eigen::MatrixXd gradient = Eigen::MatrixXd::Zero(coeffs.rows(), coeffs.cols());

			Eigen::MatrixXd hidden_z = (coeffs.block(0, 0, coeffs.rows() - 1, coeffs.cols() - 1) * input).colwise() + coeffs.rightCols<1>().head(coeffs.rows() - 1);
			Eigen::MatrixXd hidden_activation = _activation.compute(hidden_z);
			Eigen::MatrixXd reconstruction_z = (coeffs.block(0, 0, coeffs.rows() - 1, coeffs.cols() -1).transpose() * hidden_activation).colwise() + coeffs.bottomRows<1>().head(coeffs.cols() - 1).transpose();
			Eigen::MatrixXd reconstruction_activation = _activation.compute(reconstruction_z);
			Eigen::MatrixXd recontstruction_error = (reconstruction_activation - target) / input.cols();

			Eigen::MatrixXd reconstruction_delta = (coeffs.block(0, 0, coeffs.rows() - 1, coeffs.cols() - 1) * recontstruction_error)
				.cwiseProduct(_activation.gradient(hidden_z));
			Eigen::MatrixXd hidden_delta = (coeffs.block(0, 0, coeffs.rows() - 1, coeffs.cols() - 1).transpose() * reconstruction_delta)
				.cwiseProduct(_activation.gradient(input));

			// Gradients of _coefficients.transpose() and _intercept_prime
			gradient.block(0, 0, coeffs.rows() - 1, coeffs.cols() - 1) = reconstruction_delta * reconstruction_activation.transpose();
			gradient.bottomRows<1>().head(coeffs.cols() - 1) = recontstruction_error.rowwise().sum().transpose();

			// Gradients of _coefficients and _intercept
			gradient.block(0, 0, coeffs.rows() - 1, coeffs.cols() - 1) += hidden_activation * hidden_delta.transpose();
			gradient.rightCols<1>().head(coeffs.rows() - 1) = reconstruction_delta.rowwise().sum();

			gradient.block(0, 0, coeffs.rows() - 1, coeffs.cols() - 1) += _regularization * 2 * coeffs.block(0, 0, coeffs.rows() - 1, coeffs.cols() - 1);

			return gradient;
		}

		/*std::tuple<double, Eigen::MatrixXd> loss_and_gradient(const Eigen::Ref<const Eigen::MatrixXd>& coeffs, const Eigen::Ref<const Eigen::MatrixXd>& input, const Eigen::Ref<const Eigen::MatrixXd>& target) const {
			std::tuple<double, Eigen::MatrixXd> res = _activation.compute_and_gradient((coeffs.leftCols(coeffs.cols() - 1) * input).colwise() + coeffs.rightCols<1>());
			Eigen::MatrixXd gradient = Eigen::MatrixXd::Zero(coeffs.rows(), coeffs.cols());
			gradient.leftCols(coeffs.cols() - 1) = std::get<1>(res) * input.transpose() + _regularization * 2 * coeffs.leftCols(coeffs.cols() - 1);
			gradient.rightCols<1>() = std::get<1>(res).rowwise().sum();
			return{ std::get<0>(res) + _regularization * (coeffs.leftCols(coeffs.cols() - 1).array().pow(2).sum()), gradient };
		}*/

	protected:
		Eigen::MatrixXd _compute_activation(const Eigen::Ref<const Eigen::MatrixXd>& input, const Eigen::Ref<const Eigen::MatrixXd>& coefficients, const Eigen::Ref<const Eigen::VectorXd>& intercepts) const {
			return _activation.compute((coefficients * input).colwise() + intercepts);
		}

		int _hidden_units;
		Activation _activation;
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