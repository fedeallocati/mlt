#ifndef MLT_MODELS_TRANSFORMERS_ZERO_COMPONENTS_ANALYSIS_HPP
#define MLT_MODELS_TRANSFORMERS_ZERO_COMPONENTS_ANALYSIS_HPP

#include <Eigen/Core>

#include "principal_components_analysis_impl.hpp"

namespace mlt {
namespace models {
namespace transformers {
	class ZeroComponentsAnalysis : public PrincipalComponentsAnalysisImpl<ZeroComponentsAnalysis> {
	public:
		explicit ZeroComponentsAnalysis() : PrincipalComponentsAnalysisImpl(true) {}

		Eigen::MatrixXd transform(const Eigen::MatrixXd& input) const {
			assert(this->_fitted);
			return _components * PrincipalComponentsAnalysisImpl::transform(input);
		}

		Eigen::MatrixXd inverse_transform(const Eigen::MatrixXd& input) const {
			assert(this->_fitted);
			return PrincipalComponentsAnalysisImpl::inverse_transform(_components.transpose() * input);
		}
	};
}
}
}
#endif