#ifndef MULTI_LAYER_PERCEPTRON_CLASSIFIER_HPP
#define MULTI_LAYER_PERCEPTRON_CLASSIFIER_HPP

#include <Eigen/Core>
#include <Eigen/SVD>

namespace mlt {
namespace models {
namespace classifiers {
    
    // Implementation of a Multi Layer Perceptron Classifier
    // Categorization: 
    // - Application: Classifier
    // - Parametrization: Parametrized
    // - Method of Training: Gradient-Based
    // - Supervision: Supervised
	// Parameters:
	// - size_t[] hidden_layers_neurons: array with number of neurons per layer.
	// - double regularization: amount of L2 regularization to apply. Set to 0 or less if don't want to use.
	template <typename Params>
    class MultiLayerPerceptronClassifier {
    public:         
		MultiLayerPerceptronClassifier() : _init(false) {}

		MultiLayerPerceptronClassifier(size_t input, size_t classes) : _init(true) {
			size_t previous_layer = input;

			for (unsigned int i = 0; i < params_t::hidden_layers_neurons_size(); i++) {
				assert(params_t::hidden_layers_neurons(i) > 0);
				this->_theta.push_back(Eigen::MatrixXd::Random(params_t::hidden_layers_neurons(i), previous_layer + 1) * params_t::epsilon_init());

				previous_layer = params_t::hidden_layers_neurons(i);
			}

			this->_theta.push_back(Eigen::MatrixXd::Random(output, previous_layer + 1) * params_t::epsilon_init());
		}

        // Disable copy constructors
		MultiLayerPerceptronClassifier(const MultiLayerPerceptronClassifier& other) = delete;
		MultiLayerPerceptronClassifier& operator=(const MultiLayerPerceptronClassifier& other) = delete;

        inline size_t input() const {
            assert(_init);
            return this->_theta[0].rows() - 1;
        }

        inline size_t output() const {
            assert(_init);
            return this->_theta.back().cols();
        }

        inline bool add_intercept() const {
            return true;
        }

        inline bool is_initialized() const {
            return _init;
        }

        inline void init(size_t input, size_t classes) {
			size_t previous_layer = input;

			this->_theta.clear();

			for (unsigned int i = 0; i < params_t::hidden_layers_neurons_size(); i++) {
				assert(params_t::hidden_layers_neurons(i) > 0);
				this->_theta.push_back(Eigen::MatrixXd::Random(params_t::hidden_layers_neurons(i), previous_layer + 1) * params_t::epsilon_init());

				previous_layer = params_t::hidden_layers_neurons(i);
			}

			this->_theta.push_back(Eigen::MatrixXd::Random(output, previous_layer + 1) * params_t::epsilon_init());

            _init = true;
        }

        inline void reset() {
            assert(_init);
			for (auto w : this->_theta) {
				w.setRandom() * params_t::epsilon_init();
			}
        }

        inline Eigen::VectorXd score_single(const Eigen::VectorXd& input) const {
			assert(_init);
			return this->_feed_forward(this->_theta(), input.transpose()).transpose();
		}

        inline Eigen::MatrixXd score_multi(const Eigen::MatrixXd& input) const {
			assert(_init);
			return this->_feed_forward(this->_theta(), input);
        }

        inline size_t params_size() const {
            assert(_init);
			size_t counter = 0;
			for (size_t i = 0; i < this->_theta.size(); i++) {
				counter += this->_theta[i].size();
			}

			return counter;
        }

        inline Eigen::VectorXd params() const {
            assert(_init);

			size_t counter = 0;
			for (size_t i = 0; i < this->_theta.size(); i++) {
				counter += this->_theta[i].size();
			}

			Eigen::VectorXd theta_plain(counter);
			counter = 0;
			for (size_t i = 0; i < this->_theta.size(); i++) {
				theta_plain.block(counter, 0, this->_theta[i].size(), 1) = VectorXd::Map(this->_theta[i].data(), this->_theta[i].size());
				counter += this->_theta[i].size();
			}

			return theta_plain;
		}

        inline void set_params(const Eigen::VectorXd& beta) {
            assert(_init);

			size_t counter = 0;
			for (size_t i = 0; i < this->_theta.size(); i++) {
				this->_theta[i] = Eigen::MatrixXd::Map(parameters.data() + counter, this->_theta[i].rows(), this->_theta[i].cols());
				counter += this->_theta[i].size();
			}
			assert(parameters.size() == counter);
        }

        inline double cost(const Eigen::MatrixXd& input, const Eigen::MatrixXd& result) const {
            assert(_init);
            return _cost_internal(this->_theta(), input, result);
        }

        inline double cost(const Eigen::VectorXd& thetas, const Eigen::MatrixXd& input, const Eigen::MatrixXd& result) const {
			assert(_init);
			std::vector<MatrixXd> theta(this->_theta.size());

			size_t counter = 0;
			for (size_t i = 0; i < this->_theta.size(); i++) {
				theta[i] = Eigen::MatrixXd::Map(parameters.data() + counter, this->_theta[i].rows(), this->_theta[i].cols());
				counter += this->_theta[i].size();
			}

            return _cost_internal(theta, input, result);
        }
        
        inline std::tuple<double, Eigen::VectorXd> cost_and_gradient(const Eigen::MatrixXd& input, const Eigen::MatrixXd& result) const {
            assert(_init);
            auto c_a_g = _cost_and_gradient_internal(this->_theta, input, result);
			auto d_theta = std::get<1>(c_a_g);

			Eigen::VectorXd d_theta_plain(counter);
			counter = 0;
			for (size_t i = 0; i < d_theta.size(); i++) {
				d_theta_plain.block(counter, 0, d_theta[i].size(), 1) = VectorXd::Map(d_theta[i].data(), d_theta[i].size());
				counter += d_theta[i].size();
			}

			return std::make_tuple(std::get<0>(c_a_g), d_theta_plain);
        }

		inline std::tuple<double, Eigen::VectorXd> cost_and_gradient(const Eigen::VectorXd& thetas, const Eigen::MatrixXd& input, const Eigen::MatrixXd& result) const {
            assert(_init);
			std::vector<MatrixXd> theta(this->_theta.size());

			size_t counter = 0;
			for (size_t i = 0; i < this->_theta.size(); i++) {
				theta[i] = Eigen::MatrixXd::Map(parameters.data() + counter, this->_theta[i].rows(), this->_theta[i].cols());
				counter += this->_theta[i].size();
			}

            auto c_a_g = _cost_and_gradient_internal(theta, input, result);
			auto d_theta = std::get<1>(c_a_g);

			Eigen::VectorXd d_theta_plain(counter);
			counter = 0;
			for (size_t i = 0; i < d_theta.size(); i++) {
				d_theta_plain.block(counter, 0, d_theta[i].size(), 1) = VectorXd::Map(d_theta[i].data(), d_theta[i].size());
				counter += d_theta[i].size();
			}

			return std::make_tuple(std::get<0>(c_a_g), d_theta_plain);
        }

    protected:
		typedef Params::MultiLayerPerceptronClassifier params_t;

		inline MatrixXd _feed_forward(const std::vector<MatrixXd>& theta, const MatrixXd& input) const {			
			MatrixXd previous = input.transpose();

			for (unsigned int i = 0; i < theta.size() - 1; i++) {
				MatrixXd temp = MatrixXd::Ones(theta[i].rows() + 1, input.rows());
				temp.bottomRows(theta[i].rows()) = (theta[i] * previous).unaryExpr(std::ptr_fun(sigmoid));
				previous = temp;
			}

			return (theta.back() * previous).unaryExpr(std::ptr_fun(sigmoid));
		}

		inline double _cost_internal(const std::vector<Eigen::MatrixXd>& theta, const Eigen::MatrixXd& input, const Eigen::MatrixXd& result) const {
			Eigen::MatrixXd output = this->_feed_forward(theta, x);

			Eigen::MatrixXd ones = MatrixXd::Ones(theta.back().rows(), x.rows());
			double loss = ((-y).array() * output.array().log() - (ones - y).array() * (ones - output).array().log()).sum() / x.rows();
			double reg = 0;

			for (unsigned int i = 0; i < theta.size(); i++) {
				reg += theta[i].rightCols(theta[i].cols() - 1).array().pow(2).sum();
			}

			reg *= (params_t::regularization() / (double)(2 * x.rows()));

			loss += reg;

			return loss;
		}

		inline std::tuple<double, std::vector<Eigen::MatrixXd>> _cost_and_gradient_internal(const std::vector<Eigen::MatrixXd>& theta, const Eigen::MatrixXd& input, const Eigen::MatrixXd& result) const {
			std::vector<MatrixXd> z, a;
			a.push_back(input.transpose());

			for (unsigned int i = 0; i < theta.size() - 1; i++) {
				Eigen::MatrixXd temp = Eigen::MatrixXd::Ones(theta[i].rows() + 1, input.rows());
				z.push_back(theta[i] * a.back());
				temp.bottomRows(theta[i].rows()) = (z.back()).unaryExpr(std::ptr_fun(sigmoid));
				a.push_back(temp);
			}

			a.push_back((theta.back() * a.back()).unaryExpr(std::ptr_fun(sigmoid)));

			Eigen::MatrixXd ones = MatrixXd::Ones(theta.back().rows(), x.rows());
			double loss = ((-y).array() * a.back().array().log() - (ones - y).array() * (ones - a.back()).array().log()).sum() / x.rows();
			double reg = 0;

			for (unsigned int i = 0; i < theta.size(); i++) {
				reg += theta[i].rightCols(theta[i].cols() - 1).array().pow(2).sum();
			}

			reg *= (params_t::regularization() / (double)(2 * x.rows()));
			loss += reg;

			std::vector<Eigen::MatrixXd> d_theta;

			Eigen::MatrixXd previous_delta = a.back() - y;
			d_theta.push_back(previous_delta * a[a.size() - 2].transpose() / (double)y.cols());
			d_theta.back().rightCols(d_theta.back().cols() - 1) += (params_t::regularization() / (double)(x.rows())) * theta.back().rightCols(theta.back().cols() - 1);

			for (size_t i = theta.size() - 1; i > 0; i--) {
				Eigen::MatrixXd temp = (theta[i].rightCols(theta[i].cols() - 1).transpose() * previous_delta).array() *
					z[i - 1].unaryExpr(std::ptr_fun(sigmoidGradient)).array();
				d_theta.push_back(temp * a[i - 1].transpose() / (double)y.cols());
				d_theta.back().rightCols(d_theta.back().cols() - 1) += (params_t::regularization() / (double)(x.rows())) * theta[i].rightCols(theta[i].cols() - 1);
				previous_delta = temp;
			}
			
			std::reverse(d_theta.begin(), d_theta.end());

			return std::make_tuple(loss, d_theta);
        }

        bool _init;
		std::vector<Eigen::MatrixXd> _theta;
    };
}
}
}
#endif