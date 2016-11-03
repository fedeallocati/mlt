#ifndef MLT_UTILS_OPTIMIZERS_STOCHASTIC_GRADIENT_DESCENT_TRAINER_HPP
#define MLT_UTILS_OPTIMIZERS_STOCHASTIC_GRADIENT_DESCENT_TRAINER_HPP

#include <algorithm>
#include <random>
#include <tuple>
#include <vector>

#include <Eigen/Core>

#include "../../defs.hpp"
#include "../eigen.hpp"
#include "gradient_descent_updates.hpp"

namespace mlt {
namespace utils {
namespace optimizers {
	using namespace eigen;

    // Implementation of Stochastic Gradient Descent
    // Parameters:
    // - size_t batch_size: size of each batch when using mini-batches (set to 0 if using full batch gradient descent)
    // - size_t epochs: number of epochs to run the descent steps through the training data    
    // - double learning_rate: initial learning rate
    // - double learning_rate_decay: learning_rate is multiplied by this after each epoch
    // - UpdateMethod update_method: object implementing the update strategy to use
	template <class UpdateMethod = RMSPropGradientDescentUpdate>
    class StochasticGradientDescent {
    public:
		StochasticGradientDescent(size_t batch_size = 1, size_t epochs = 1000, double learning_rate = 0.001, double learning_rate_decay = 0.99,
			const UpdateMethod& update_method = UpdateMethod()) : _batch_size(batch_size), _epochs(epochs), _learning_rate(learning_rate), 
			_current_learning_rate(learning_rate), _learning_rate_decay(learning_rate_decay), _update_method(update_method) {}

        template <class Model, class Target>
		auto operator()(const Model& model, Features input, Target target, MatrixXdRef init, bool cold_start) {
			assert(input.cols() == target.cols());

            if (cold_start) {
                _current_learning_rate = _learning_rate;
				_update_method.restart();
            }

            auto iters_per_epoch = _batch_size > 0 && _batch_size <= input.cols() ? input.cols() / _batch_size : 1;

			MatrixXd params = init;

			random_device rd;
            default_random_engine generator(rd());

            for (auto epoch = 0; epoch < _epochs; epoch++) {
                for (auto iter = 0; iter < iters_per_epoch; iter++) {
					if (_batch_size > 0) {
						auto subset = tied_random_cols_subset(input, target, _batch_size, generator);
						auto input_batch = get<0>(subset);
						auto target_batch = get<1>(subset);

						params += _update_method.step(_current_learning_rate, model.gradient(params, input_batch, target_batch));
					} else {
						params += _update_method.step(_current_learning_rate, model.gradient(params, input, target));
					}
                }

#ifdef MLT_VERBOSE
				if (epoch % 10 == 0) {
					auto loss = model.loss(params, input, target);
					MLT_LOG_LINE("[SGD]: Epoch " << epoch + 1 << "/" << _epochs << " Loss " << loss);
				}
#endif

				_current_learning_rate *= _learning_rate_decay;
            }

			return params;
        }

	protected:
		size_t _batch_size;
		size_t _epochs;
		double _learning_rate;
		double _current_learning_rate;
		double _learning_rate_decay;
		UpdateMethod _update_method;
    };
}
}
}
#endif