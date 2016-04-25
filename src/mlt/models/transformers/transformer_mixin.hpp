#ifndef MLT_MODELS_TRANSFORMERS_TRANSFORMER_MIXIN_HPP
#define MLT_MODELS_TRANSFORMERS_TRANSFORMER_MIXIN_HPP

#include <Eigen/Core>

namespace mlt {
namespace models {
namespace transformers {
	template <class Transformer>
	class TransformerMixin {
	public:
		Transformer& fit(const Eigen::MatrixXd& input, bool cold_start = true) {
			return static_cast<Transformer&>(*this).fit(input, cold_start);
		}

		Transformer& fit(const Eigen::MatrixXd& input, const Eigen::MatrixXd&, bool cold_start = true) {
			return static_cast<Transformer&>(*this).fit(inputcold_start);
		}

		Transformer& fit(const Eigen::MatrixXd& input, const Eigen::MatrixXi&, bool cold_start = true) {
			return static_cast<Transformer&>(*this).fit(inputcold_start);
		}

		Eigen::MatrixXd fit_transform(const Eigen::MatrixXd& input, bool cold_start = true) {
			return static_cast<Transformer&>(*this).fit(input, cold_start).transform(input);
		}

		Eigen::MatrixXd fit_transform(const Eigen::MatrixXd& input, const Eigen::MatrixXd&, bool cold_start = true) {
			return static_cast<Transformer&>(*this).fit(input, cold_start).transform(input);
		}

		Eigen::MatrixXd fit_transform(const Eigen::MatrixXd& input, const Eigen::MatrixXi&, bool cold_start = true) {
			return static_cast<Transformer&>(*this).fit(input, cold_start).transform(input);
		}
	};
}
}
}
#endif