#ifndef MLT_MODELS_LINEAR_MODEL_HPP
#define MLT_MODELS_LINEAR_MODEL_HPP

#include <Eigen/Core>

#include "base.hpp"
#include "../utils/linear_algebra.hpp"

namespace mlt {
namespace models {
	using namespace utils::linear_algebra;

	template <class BaseModelType>
	class LinearModel : public BaseModelType {
	public:
		inline auto fit_intercept() const { return _fit_intercept; }

		inline const auto coefficients() const { assert(_fitted); return _fit_intercept ? _coefficients.leftCols(_coefficients.cols() - 1).eval() : _coefficients; }

		inline const auto intercepts() const { assert(_fitted && _fit_intercept); return _coefficients.rightCols<1>().eval(); }

		inline const auto all_coefficients() const { assert(_fitted); return _coefficients; }

	protected:
		LinearModel(bool fit_intercept) : _fit_intercept(fit_intercept) {}

		inline void _set_coefficients(MatrixXdRef coefficients) {
			_coefficients = coefficients;
			_fitted = true;
		}

		inline auto _apply_linear_transformation(Features input) const {
			assert(_fitted);

			if (_fit_intercept) {
				return linear_transformation(input, _coefficients.leftCols(_coefficients.cols() - 1), _coefficients.rightCols<1>());
			}

			return linear_transformation(input, _coefficients);
		}

		inline auto _apply_linear_transformation(Features input, MatrixXdRef coeffs) const {
			if (_fit_intercept) {
				return linear_transformation(input, coeffs.leftCols(coeffs.cols() - 1), coeffs.rightCols<1>());
			}
			
			return linear_transformation(input, coeffs);
		}

		const bool _fit_intercept;

	private:
		MatrixXd _coefficients;
	};
}
}

#endif