#ifndef MLT_MODELS_TRANSFORMERS_PRINCIPAL_COMPONENTS_ANALYSIS_HPP
#define MLT_MODELS_TRANSFORMERS_PRINCIPAL_COMPONENTS_ANALYSIS_HPP

#include "principal_components_analysis_impl.hpp"

namespace mlt {
namespace models {
namespace transformers {
	class PrincipalComponentsAnalysis : public PrincipalComponentsAnalysisImpl<PrincipalComponentsAnalysis> {
	public:
		using PrincipalComponentsAnalysisImpl<PrincipalComponentsAnalysis>::PrincipalComponentsAnalysisImpl;
	};
}
}
}
#endif