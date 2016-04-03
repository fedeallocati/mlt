#ifndef MLT_MODELS_LINEAR_MODEL_HPP
#define MLT_MODELS_LINEAR_MODEL_HPP

#include <Eigen/Core>

#include "base_model.hpp"

namespace mlt {
namespace models {

	class LinearModel : public BaseModel {
	public:
		bool fit_intercept() const { return _fit_intercept; }

		Eigen::MatrixXd coefficients() const { return _coefficients; }

		Eigen::VectorXd intercepts() const { return _intercepts.transpose(); }

		Eigen::MatrixXd predict(const Eigen::MatrixXd& input) const {
			assert(_fitted);

			Eigen::MatrixXd res;

			if (input.cols() == 1 && input.rows() == _coefficients.rows()) {
				// input = 5 * 1
				// coefficients = 5 * 3
				// intercepts = 3
				// res = 3 * 1;
				res = _coefficients.transpose() * input;

				if (fit_intercept)
					res += _intercepts;
			}
			else {
				// input = 10 * 5
				// coefficients = 5 * 3
				// intercepts = 3
				// res = 10 * 3
				res = input * _coefficients;
				if (fit_intercept)
					res = res.rowwise() + _intercepts;
			}
			
			return res;
		}

	protected:
		explicit LinearModel(bool fit_intercept) : _fit_intercept(fit_intercept) {}

		void _set_coefficients(const Eigen::MatrixXd& coefficients) {
			assert(!_fit_intercept);

			_coefficients = coefficients;

			_fitted = true;
			_input_size = coefficients.rows();
			_output_size = coefficients.cols();
		}

		void _set_coefficients_and_intercepts(const Eigen::MatrixXd& coefficients, const Eigen::RowVectorXd& intercepts) {
			assert(coefficients.cols() == intercepts.size());

			_coefficients = coefficients;
			_intercepts = intercepts;

			_fitted = true;
			_input_size = coefficients.rows();
			_output_size = coefficients.cols();
		}

		const bool _fit_intercept = false;

	private:
		Eigen::MatrixXd _coefficients;
		Eigen::RowVectorXd _intercepts;

	};
}
}

#endif