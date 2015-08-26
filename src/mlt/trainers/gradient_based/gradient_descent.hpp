#ifndef GRADIENT_DESCENT_TRAINER_HPP
#define GRADIENT_DESCENT_TRAINER_HPP

#include <vector>
#include <algorithm>
#include <random>

#include <Eigen/Core>

namespace mlt {
namespace trainers {
namespace gradient_based {
    
    // Implementation of Gradient Descent Trainer
    // Parameters:
	// - int epochs: number of epochs to run the descent steps through the training data
	// - int batch_size: size of each batch when using mini-batches (set to 0 if using full batch gradient descent)
	// - double learning_rate: initial learning rate
	// - double learning_rate_decay: learning_rate is multiplied by this after each epoch
	// - double momentum: the amount of momentum applied on each descent (set to 0 if don't want to use momentum)
	// - bool nesterov_momentum: indicates whether to use Nesterov's Accelerated Momentum or normal Momentum update rule	
	template <typename Params, typename Model>
    class GradientDescentTrainer {
	public:
		typedef Model model_t;
		GradientDescentTrainer(model_t& model) : _model(model), _epochs(0), _velocity(Eigen::VectorXd::Zero(model.params_size())) {}

		// Disable copy constructors
		GradientDescentTrainer(const GradientDescentTrainer& other) = delete;
		GradientDescentTrainer& operator=(const GradientDescentTrainer& other) = delete;

		inline const model_t& model() const {
			return _model;
		}

		template<typename T>
		std::enable_if<std::is_convertible<T, Eigen::MatrixXd>::value || std::is_convertible<T, Eigen::MatrixXi>::value, void>::type
		train(const Eigen::MatrixXd& input, const T& result, bool reset = false) {
			if (reset) {
				_model.reset();
				_epochs = 0;
				_velocity = Eigen::VectorXd::Zero(_model.params_size());
			}

			auto iters_per_epoch = params_t::batch_size > 0 && params_t::batch_size <= input.rows() ? input.rows() / params_t::batch_size : 1;

			Eigen::VectorXd params = _model.params();
			auto learning_rate = params_t::learning_rate * std::pow(params_t::learning_rate_decay, _epochs);

			std::default_random_engine generator;
			std::uniform_int_distribution<int> distribution(0, input.rows() - 1);

			for (auto epoch = 0; epoch < params_t::epochs; epoch++) {
				for (auto iter = 0; iter < iters_per_epoch; iter++) {
					Eigen::MatrixXd input_batch;
					T result_batch;
					if (iters_per_epoch == 1) {
						input_batch = input;
						result_batch = result;
					} else {
						input_batch = Eigen::MatrixXd(params_t::batch_size, input.cols());
						result_batch = Eigen::MatrixXd(params_t::batch_size, result.cols());						
						std::vector<int> indexs(params_t::batch_size);
						std::generate(indexs.begin(), indexs.end(), [&]() { return distribution(generator); });

						auto i = 0;
						for (auto ridx : indexs) {							
							input_batch.row(i) = input.row(ridx);
							result_batch.row(i) = result.row(ridx);
							i++;
						}
					}

					Eigen::VectorXd grad = _model.cost_gradient(params, input_batch, result_batch);
					VectorXd velocity_prev = _velocity;

					if (params_t::momentum > 0) {
						_velocity = params_t::momentum * _velocity - learning_rate * grad;
					} else {
						_velocity = - learning_rate * grad;
					}

					if (params_t::nesterov_momentum && params_t::momentum > 0) {
						params += -params_t::momentum * velocity_prev + (1 + params_t::momentum) * _velocity;
					} else {						
						params += _velocity;
					}					
				}

				learning_rate *= params_t::learning_rate_decay;
			}

			_epochs += params_t::epochs;
			_model.set_params(params);
		}

	protected:
		typedef Params::GradientDescent params_t;
		model_t& _model;
		size_t _epochs;
		Eigen::VectorXd _velocity;
	};
}
}
}
#endif