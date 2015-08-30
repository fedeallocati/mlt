#ifndef GRADIENT_DESCENT_TRAINER_HPP
#define GRADIENT_DESCENT_TRAINER_HPP

#include <vector>
#include <algorithm>
#include <random>

#include <Eigen/Core>

namespace mlt {
namespace trainers {
namespace gradient_based {

    enum gradient_descent_update_t {
        gradient_descent = 0,
        momentum,
        nesterov_momentum,
        adagrad,
        rmsprop
    };
    
    // Implementation of Gradient Descent Trainer
    // Parameters:
    // - int epochs: number of epochs to run the descent steps through the training data
    // - int batch_size: size of each batch when using mini-batches (set to 0 if using full batch gradient descent)
    // - double learning_rate: initial learning rate
    // - double learning_rate_decay: learning_rate is multiplied by this after each epoch
    // - gradient_descent_update_t update_method: indicates what is the update rule to use
    // - double update_param: if update_method is momentum or nesterov_momentum the amount of momentum applied on each
    //                        descent; if it is rmsprop, the decay_rate parameter   
    template <typename Params, typename Model>
    class GradientDescentTrainer {
    public:
        typedef Model model_t;
        GradientDescentTrainer(model_t& model) : _model(model), _epochs(0), _update_method_cache(Eigen::VectorXd::Zero(model.params_size())) {}

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
                _update_method_cache = Eigen::VectorXd::Zero(_model.params_size());
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

                    Eigen::VectorXd grad = std::get<1>(_model.cost_and_gradient(params, input_batch, result_batch));
                    switch (params_t::update_method) {
                    case gradient_descent_update_t::gradient_descent:
                        params += -learning_rate * grad;
                        break;
                    case gradient_descent_update_t::momentum:
                        _update_method_cache = params_t::update_param * _update_method_cache - learning_rate * grad;
                        params += _update_method_cache;
                        break;
                    case gradient_descent_update_t::nesterov_momentum:
                    {
                        VectorXd velocity_prev = _update_method_cache;
                        _update_method_cache = params_t::update_param * _update_method_cache - learning_rate * grad;
                        params += -params_t::update_param * velocity_prev + (1 + params_t::update_param) * _update_method_cache;
                        break;
                    }
                    case gradient_descent_update_t::adagrad:
                        _update_method_cache += grad.array().pow(2).matrix();
                        params += -learning_rate * (grad.array() / (_update_method_cache.array() + 1e-8).sqrt()).matrix();
                        break;
                    case gradient_descent_update_t::rmsprop:
                        _update_method_cache = params_t::update_param * _update_method_cache + (1 - params_t::update_param) * grad.array().pow(2).matrix();                     
                        params += -learning_rate * (grad.array() / (_update_method_cache.array() + 1e-8).sqrt()).matrix();
                        break;
                    }
                }

                learning_rate *= params_t::learning_rate_decay;

                #ifdef MLT_VERBOSE_TRAINING
                std::cout << "Finished epoch " << epoch << "/" << params_t::epochs << ": cost " << _model.cost(params, input, result) << endl;
                #endif
            }

            _epochs += params_t::epochs;
            _model.set_params(params);
        }

    protected:
        typedef Params::GradientDescent params_t;

        model_t& _model;
        size_t _epochs;
        Eigen::VectorXd _update_method_cache;
    };
}
}
}
#endif