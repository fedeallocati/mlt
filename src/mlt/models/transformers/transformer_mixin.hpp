#ifndef MLT_MODELS_TRANSFORMERS_TRANSFORMER_MIXIN_HPP
#define MLT_MODELS_TRANSFORMERS_TRANSFORMER_MIXIN_HPP

#include <Eigen/Core>
#include <vector>

namespace mlt {
namespace models {
namespace transformers {
	template <class Transformer>
	class TransformerMixin {
	public:
		template <typename Input>
		Transformer& fit(Input&& input, const Eigen::Ref<const Eigen::MatrixXd>&, bool cold_start = true) {
			return static_cast<Transformer&>(*this).fit(std::forward<Input>(input), cold_start);
		}

		template <typename Input>
		Transformer& fit(Input&& input, const Eigen::Ref<const Eigen::VectorXi>&, bool cold_start = true) {
			return static_cast<Transformer&>(*this).fit(std::forward<Input>(input), cold_start);
		}

		template <typename Input>
		Eigen::MatrixXd fit_transform(Input&& input, const Eigen::Ref<const Eigen::MatrixXd>&, bool cold_start = true) {
			return static_cast<Transformer&>(*this).fit(std::forward<Input>(input), cold_start).transform(input);
		}

		template <typename Input>
		Eigen::MatrixXd fit_transform(Input&& input, const Eigen::Ref<const Eigen::VectorXi>&, bool cold_start = true) {
			return static_cast<Transformer&>(*this).fit(std::forward<Input>(input), cold_start).transform(input);
		}

	protected:
		TransformerMixin() {}
	};
}
}
}
#endif