#ifndef MLT_MODELS_BASE_HPP
#define MLT_MODELS_BASE_HPP

#include <Eigen/Core>

#include "../defs.hpp"

namespace mlt {
	template <class ConcreteType>
	class Model {
	public:
		using Self = ConcreteType;

		inline bool fitted() const { return _fitted; }

	protected:
		Model() = default;
		Model(const Model&) = default;
		Model(Model&&) = default;
		Model& operator=(const Model&) = default;
		~Model() = default;

		const Self& _self() const { return static_cast<const Self&>(*this); }

		Self& _self() { return static_cast<Self&>(*this); }

		bool _fitted = false;

	};

	template <class ConcreteType, class Output>
	class Predictor : public Model<ConcreteType> {
	public:
		using Target = const Ref<const Output>&;
		using Result = Output;

		inline double score(Features input, Target target) {
			assert(false);
		}

		inline MatrixXd to_target_matrix(Target target) {
			assert(false);
		}

	protected:
		Predictor() = default;
		Predictor(const Predictor&) = default;
		Predictor(Predictor&&) = default;
		Predictor& operator=(const Predictor&) = default;
		~Predictor() = default;
	};
}
#endif