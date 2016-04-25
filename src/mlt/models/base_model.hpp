#ifndef MLT_MODELS_BASE_MODEL_HPP
#define MLT_MODELS_BASE_MODEL_HPP

namespace mlt {
namespace models {
	class BaseModel {
	public:
		bool fitted() const { return _fitted; }

		size_t input_size() const { assert(fitted); return _input_size; }

		size_t output_size() const { assert(fitted); return _output_size; }

	protected:
		BaseModel() = default;
		BaseModel(const BaseModel&) = default;
		BaseModel(BaseModel&&) = default;
		BaseModel& operator=(const BaseModel&) = default;
		~BaseModel() = default;

		bool _fitted = false;
		size_t _input_size = -1;
		size_t _output_size = -1;
	};
}
}

#endif