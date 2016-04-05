#ifndef MLT_MODELS_LINEAR_MODEL_HPP
#define MLT_MODELS_LINEAR_MODEL_HPP

#include <Eigen/Core>

#include "base_model.hpp"

namespace mlt {
namespace models {

	class LinearModel : public BaseModel {
	public:
		bool fit_intercept() const { return _fit_intercept; }

		Eigen::MatrixXd coefficients() const { assert(_fitted); return _coefficients_transposed.transpose(); }

		const Eigen::VectorXd& intercepts() const { assert(_fitted && _fit_intercept); return _intercepts; }

	protected:
		explicit LinearModel(bool fit_intercept) : _fit_intercept(fit_intercept) {}

		void _set_coefficients(const Eigen::MatrixXd& coefficients) {
			assert(!_fit_intercept);

			_coefficients_transposed = coefficients.transpose();

			_fitted = true;
			_input_size = coefficients.cols();
			_output_size = coefficients.rows();
		}

		void _set_coefficients_and_intercepts(const Eigen::MatrixXd& coefficients, const Eigen::VectorXd& intercepts) {
			assert(coefficients.rows() == intercepts.rows());

			_coefficients_transposed = coefficients.transpose();
			_intercepts = intercepts;

			_fitted = true;
			_input_size = coefficients.cols();
			_output_size = coefficients.rows();
		}

		Eigen::MatrixXd _apply_linear_transformation(const Eigen::MatrixXd& input) const {
			assert(_fitted);

			Eigen::MatrixXd res;

			if (fit_intercept) {
				return (_coefficients_transposed * input).colwise() + _intercepts;
			}

			return _coefficients_transposed * input;
		}

		const bool _fit_intercept = false;

	private:
		Eigen::MatrixXd _coefficients_transposed;
		Eigen::VectorXd _intercepts;
	};
}
}

#endif