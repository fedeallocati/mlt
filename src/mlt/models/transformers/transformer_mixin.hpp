#ifndef MLT_MODELS_TRANSFORMERS_TRANSFORMER_MIXIN_HPP
#define MLT_MODELS_TRANSFORMERS_TRANSFORMER_MIXIN_HPP

#include <Eigen/Core>

namespace mlt {
namespace models {
namespace transformers {
	template <class Transformer>
	class TransformerMixin {
	public:
		template <typename Input, typename Target>
		Transformer& fit(Input&& input, Target&&, bool cold_start = true) {
			return static_cast<Transformer&>(*this).fit(std::forward<Input>(input), cold_start);
		}

		template <typename Input, typename Target>
		Eigen::MatrixXd fit_transform(Input&& input, Target&&, bool cold_start = true) {
			return static_cast<Transformer&>(*this).fit(std::forward<Input>(input), cold_start).transform(input);
		}
	};
}
}
}
#endif