#ifndef MLT_MODELS_TRANSFORMATIONS_ZERO_COMPONENTS_ANALYSIS_HPP
#define MLT_MODELS_TRANSFORMATIONS_ZERO_COMPONENTS_ANALYSIS_HPP

#include "principal_components_analysis.hpp"

namespace mlt {
namespace models {
namespace transformations {
	class ZeroComponentsAnalysis : public PrincipalComponentsAnalysis {
	public:
		explicit ZeroComponentsAnalysis() : PrincipalComponentsAnalysis(true) {}
		
		Eigen::MatrixXd transform(const Eigen::MatrixXd& input) const {
			assert(_fitted);

			return _components * PrincipalComponentsAnalysis::transform(input);
		}

		Eigen::MatrixXd inverse_transform(const Eigen::MatrixXd& input) const {
			assert(_fitted);

			return PrincipalComponentsAnalysis::inverse_transform(_components.transpose() * input);
		}
	};
}
}
}
#endif