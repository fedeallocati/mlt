#ifndef MLT_MODELS_REGRESSORS_REGRESSOR_MIXIN_HPP
#define MLT_MODELS_REGRESSORS_REGRESSOR_MIXIN_HPP

#include <Eigen/Core>

namespace mlt {
namespace models {
namespace regressors {
	template <class Regressor>
	class RegressorMixin {
	public:
		double score(const Eigen::Ref<const Eigen::MatrixXd>& input, const Eigen::Ref<const Eigen::MatrixXd>& target) const {
			return (1 - ((y - this->_concrete().predict(input)).array().pow(2).rowwise().sum().array() /
				(y.colwise() - (y.rowwise().mean())).array().pow(2).rowwise().sum().array())).mean();
		}

	protected:
		RegressorMixin() {}

	private:
		const Regressor& _concrete() const { return static_cast<const Regressor&>(*this); }

		Regressor& _concrete() { return static_cast<Regressor&>(*this); }
	};
}
}
}
#endif