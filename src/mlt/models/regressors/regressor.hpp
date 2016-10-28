#ifndef MLT_MODELS_REGRESSORS_REGRESSOR_HPP
#define MLT_MODELS_REGRESSORS_REGRESSOR_HPP

#include <Eigen/Core>

#include "../base.hpp"

namespace mlt {
namespace models {
namespace regressors {
	template <class ConcreteType>
	class Regressor : public Predictor<ConcreteType, MatrixXd> {
	public:
		inline auto score(Features input, Target target) const {
			return (1 - ((y - _self().predict(input)).array().pow(2).rowwise().sum().array() / (y.colwise() - (y.rowwise().mean())).array().pow(2).rowwise().sum().array())).mean();
		}

	protected:
		Regressor() = default;
		Regressor(const Regressor&) = default;
		Regressor(Regressor&&) = default;
		Regressor& operator=(const Regressor&) = default;
		~Regressor() = default;

		inline auto _to_target_matrix(Target target) {
			return target;
		}
	};
}
}
}
#endif