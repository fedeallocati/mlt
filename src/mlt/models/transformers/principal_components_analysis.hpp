#ifndef MLT_MODELS_TRANSFORMERS_PRINCIPAL_COMPONENTS_ANALYSIS_HPP
#define MLT_MODELS_TRANSFORMERS_PRINCIPAL_COMPONENTS_ANALYSIS_HPP

#include "principal_components_analysis_impl.hpp"

namespace mlt {
namespace models {
namespace transformers {
	class PrincipalComponentsAnalysis : public PrincipalComponentsAnalysisImpl<PrincipalComponentsAnalysis> {
	public:
		explicit PrincipalComponentsAnalysis(int components_size, bool whiten = false) : PrincipalComponentsAnalysisImpl(components_size, whiten) {}

		explicit PrincipalComponentsAnalysis(double variance_to_retain, bool whiten = false) : PrincipalComponentsAnalysisImpl(variance_to_retain, whiten) {
			assert(variance_to_retain > 0 && variance_to_retain <= 1);
		}

		explicit PrincipalComponentsAnalysis(bool whiten = false) : PrincipalComponentsAnalysisImpl(whiten) {}
	};
}
}
}
#endif