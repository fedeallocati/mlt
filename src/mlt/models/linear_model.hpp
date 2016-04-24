#ifndef MLT_MODELS_LINEAR_MODEL_HPP
#define MLT_MODELS_LINEAR_MODEL_HPP

#include <Eigen/Core>

#include "base_model.hpp"
#include <iostream>

namespace mlt {
namespace models {

	class LinearModel : public BaseModel {
	public:
		bool fit_intercept() const { return _fit_intercept; }

		const Eigen::MatrixXd& coefficients() const { assert(_fitted); return _coefficients; }

		const Eigen::VectorXd& intercepts() const { assert(_fitted && _fit_intercept); return _intercepts; }

	protected:
		explicit LinearModel(bool fit_intercept) : _fit_intercept(fit_intercept) {}

		void _set_coefficients(const Eigen::Ref<const Eigen::MatrixXd>& coefficients) {
			assert(!_fit_intercept);

			_coefficients = coefficients;

			_fitted = true;
			_input_size = coefficients.rows();
			_output_size = coefficients.cols();
		}

		void _set_coefficients_and_intercepts(const Eigen::Ref<const Eigen::MatrixXd>& coefficients, const Eigen::Ref<const Eigen::VectorXd>& intercepts) {
			assert(_fit_intercept);
			assert(coefficients.rows() == intercepts.rows());

			_coefficients = coefficients;
			_intercepts = intercepts;

			_fitted = true;
			_input_size = coefficients.rows();
			_output_size = coefficients.cols();
		}

		Eigen::MatrixXd _apply_linear_transformation(const Eigen::Ref<const Eigen::MatrixXd>& input) const {
			assert(_fitted);

			if (_fit_intercept) {
				return _apply_linear_transformation(input, _coefficients, _intercepts);
			}

			return _apply_linear_transformation(input, _coefficients);
		}

		static Eigen::MatrixXd _apply_linear_transformation(const Eigen::Ref<const Eigen::MatrixXd>& input, const Eigen::Ref<const Eigen::MatrixXd>& coefficients) {
			return coefficients * input;
		}

		static Eigen::MatrixXd _apply_linear_transformation(const Eigen::Ref<const Eigen::MatrixXd>& input, const Eigen::Ref<const Eigen::MatrixXd>& coefficients, const Eigen::Ref<const Eigen::VectorXd>& intercepts) {
			return (coefficients * input).colwise() + intercepts;
		}

		const bool _fit_intercept;

	private:
		
		Eigen::MatrixXd _coefficients;
		Eigen::VectorXd _intercepts;
	};
}
}

#endif