#ifndef MLT_UTILS_OPTIMIZERS_STOCHASTIC_GRADIENT_DESCENT_TRAINER_HPP
#define MLT_UTILS_OPTIMIZERS_STOCHASTIC_GRADIENT_DESCENT_TRAINER_HPP

#include <vector>
#include <algorithm>
#include <random>

#include <Eigen/Core>

#include "gradient_descent_updates.hpp"
#include "../eigen.hpp"

namespace mlt {
namespace utils {
namespace optimizers {
    // Implementation of Stochastic Gradient Descent
    // Parameters:
    // - int epochs: number of epochs to run the descent steps through the training data
    // - int batch_size: size of each batch when using mini-batches (set to 0 if using full batch gradient descent)
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
		typename Eigen::MatrixXd run(const Model& model, const Eigen::MatrixXd& input, const Target& target, const Eigen::Ref<const Eigen::MatrixXd>& init, bool cold_start) {
            if (cold_start) {
                _current_learning_rate = _learning_rate;
				_update_method.restart();
            }

            auto iters_per_epoch = _batch_size > 0 && _batch_size <= input.cols() ? input.cols() / _batch_size : 1;

			Eigen::VectorXd params = eigen::ravel(init);

            std::default_random_engine generator;
            std::uniform_int_distribution<int> distribution(0, input.cols() - 1);

            for (auto epoch = 0; epoch < _epochs; epoch++) {
                for (auto iter = 0; iter < iters_per_epoch; iter++) {
                    Eigen::MatrixXd input_batch = Eigen::MatrixXd(input.rows(), _batch_size);
                    Target target_batch = Eigen::MatrixXd(target.rows(), _batch_size);

                    std::vector<int> indexs(_batch_size);
                    std::generate(indexs.begin(), indexs.end(), [&]() { return distribution(generator); });

                    auto i = 0;
                    for (auto ridx : indexs) {
                        input_batch.col(i) = input.col(ridx);
						target_batch.col(i) = target.col(ridx);
                        i++;
                    }

                    auto gradient = eigen::ravel(std::get<1>(model.loss_and_gradient(eigen::unravel(params, init.rows(), init.cols()), input_batch, target_batch)));
					params += _update_method.step(_current_learning_rate, gradient);
                }

				_current_learning_rate *= _learning_rate_decay;

                std::cout << "Finished epoch " << epoch + 1 << "/" << _epochs << ": cost " << model.loss(eigen::unravel(params, init.rows(), init.cols()), input, target) << std::endl;
            }

			return eigen::unravel(params, init.rows(), init.cols());
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