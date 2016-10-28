#ifndef MLT_MODELS_TRANSFORMERS_TRANSFORMER_HPP
#define MLT_MODELS_TRANSFORMERS_TRANSFORMER_HPP

#include <type_traits>

#include <Eigen/Core>

#include "../base.hpp"

namespace mlt {
namespace models {
namespace transformers {
	template <class ConcreteType>
	class Transformer : public Model<ConcreteType> {
	public:
		using Result = MatrixXd;
		template <class FitInput, class FitTarget>
		Self& fit(FitInput input, FitTarget, bool cold_start = true) {
			return _self().fit(forward<FitInput>(input), cold_start);
		}

		template <class FitInput, class FitTarget>
		Result fit_transform(FitInput&& input, FitTarget, bool cold_start = true) {
			return _self().fit(forward<FitInput>(input), cold_start).transform(input);
		}

	protected:
		Transformer() = default;
		Transformer(const Transformer&) = default;
		Transformer(Transformer&&) = default;
		Transformer& operator=(const Transformer&) = default;
		~Transformer() = default;
	};
}
}
}
#endif