#ifndef MLT_MODELS_TRANSFORMERS_ZERO_COMPONENTS_ANALYSIS_HPP
#define MLT_MODELS_TRANSFORMERS_ZERO_COMPONENTS_ANALYSIS_HPP

#include "principal_components_analysis_impl.hpp"

namespace mlt {
namespace models {
namespace transformers {
	class ZeroComponentsAnalysis : public PrincipalComponentsAnalysisImpl<ZeroComponentsAnalysis> {
	public:
		explicit ZeroComponentsAnalysis() : PrincipalComponentsAnalysisImpl(true) {}

		Result transform(Features input) const {
			assert(_fitted);
			return _components * PrincipalComponentsAnalysisImpl<Self>::transform(input);
		}

		Features inverse_transform(Result input) const {
			assert(_fitted);
			return PrincipalComponentsAnalysisImpl::inverse_transform(_components.transpose() * input);
		}
	};
}
}
}
#endif