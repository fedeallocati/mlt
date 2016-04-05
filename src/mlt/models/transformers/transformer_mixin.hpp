#ifndef MLT_MODELS_TRANSFORMERS_TRANSFORMER_MIXIN_HPP
#define MLT_MODELS_TRANSFORMERS_TRANSFORMER_MIXIN_HPP

#include <Eigen/Core>

namespace mlt {
namespace models {
namespace transformers {
	template <class Transformer>
	class TransformerMixin {
	public:
		Transformer& fit(const Eigen::MatrixXd& input) {
			return static_cast<Transformer&>(*this).fit(input);
		}

		Transformer& fit(const Eigen::MatrixXd& input, const Eigen::MatrixXd&) {
			return static_cast<Transformer&>(*this).fit(input);
		}

		Transformer& fit(const Eigen::MatrixXd& input, const Eigen::MatrixXi&) {
			return static_cast<Transformer&>(*this).fit(input);
		}

		Eigen::MatrixXd fit_transform(const Eigen::MatrixXd& input) {
			return static_cast<Transformer&>(*this).fit(input).transform(input);
		}

		Eigen::MatrixXd fit_transform(const Eigen::MatrixXd& input, const Eigen::MatrixXd&) {
			return static_cast<Transformer&>(*this).fit(input).transform(input);
		}

		Eigen::MatrixXd fit_transform(const Eigen::MatrixXd& input, const Eigen::MatrixXi&) {
			return static_cast<Transformer&>(*this).fit(input).transform(input);
		}
	};
}
}
}
#endif