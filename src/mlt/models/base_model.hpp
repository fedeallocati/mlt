#ifndef MLT_MODELS_BASE_MODEL_HPP
#define MLT_MODELS_BASE_MODEL_HPP

namespace mlt {
namespace models {

	class BaseModel {
	public:
		bool fitted() const { return _fitted; }

		size_t input_size() const { return _input_size; }

		size_t output_size() const { return _output_size; }
	protected:
		BaseModel() {};

		bool _fitted = false;
		size_t _input_size = -1;
		size_t _output_size = -1;
	};
}
}

#endif