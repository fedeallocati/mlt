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
			std::cout << "TransformerMixin fit" << std::endl;
			return static_cast<Transformer&>(*this).fit(input);
		}

		Transformer& fit(const Eigen::MatrixXd& input, const Eigen::MatrixXd&) {
			std::cout << "TransformerMixin fit" << std::endl;
			return static_cast<Transformer&>(*this).fit(input);
		}

		Transformer& fit(const Eigen::MatrixXd& input, const Eigen::MatrixXi&) {
			std::cout << "TransformerMixin fit" << std::endl;
			return static_cast<Transformer&>(*this).fit(input);
		}
	};
}
}
}
#endif