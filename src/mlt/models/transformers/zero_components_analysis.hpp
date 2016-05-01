#ifndef MLT_MODELS_TRANSFORMERS_ZERO_COMPONENTS_ANALYSIS_HPP
#define MLT_MODELS_TRANSFORMERS_ZERO_COMPONENTS_ANALYSIS_HPP

#include <Eigen/Core>

#include "principal_components_analysis_impl.hpp"
#include "transformer_mixin.hpp"

namespace mlt {
namespace models {
namespace transformers {
	class ZeroComponentsAnalysis : public PrincipalComponentsAnalysisImpl, public TransformerMixin<ZeroComponentsAnalysis> {
	public:
		explicit ZeroComponentsAnalysis() : PrincipalComponentsAnalysisImpl(true) {}

		using TransformerMixin<ZeroComponentsAnalysis>::fit;

		ZeroComponentsAnalysis& fit(const Eigen::Ref<const Eigen::MatrixXd>& input, bool = true)
		{
			this->_fit(input);
			return *this;
		}

		Eigen::MatrixXd transform(const Eigen::Ref<const Eigen::MatrixXd>& input) const {
			assert(this->_fitted);
			return _components * PrincipalComponentsAnalysisImpl::transform(input);
		}

		Eigen::MatrixXd inverse_transform(const Eigen::Ref<const Eigen::MatrixXd>& input) const {
			assert(this->_fitted);
			return PrincipalComponentsAnalysisImpl::inverse_transform(_components.transpose() * input);
		}
	};
}
}
}
#endif