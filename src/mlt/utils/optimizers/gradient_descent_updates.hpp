#ifndef MLT_UTILS_OPTIMIZERS_GRADIENT_DESCENT_UPDATES_HPP
#define MLT_UTILS_OPTIMIZERS_GRADIENT_DESCENT_UPDATES_HPP

#include <Eigen/Core>

#include "../../defs.hpp"

namespace mlt {
namespace utils {
namespace optimizers {
	// Implementation of the Vanilla Gradient Descent update rule
	class VanillaGradientDescentUpdate {
	public:
		void restart() {}

		auto step(double learning_rate, MatrixXdRef gradient) {
			return (-learning_rate * gradient).eval();
		}
	};

	// Implementation of the Momentum Gradient Descent update rule
	// Parameters:
	// - double mu: amount of momentum applied on each descent
	class MomentumGradientDescentUpdate {
	public:
		MomentumGradientDescentUpdate(double mu = 0.9) : _mu(mu), _init(false) {}

		void restart() { _init = false; }

		auto step(double learning_rate, MatrixXdRef gradient) {
			if (!_init || _cache.rows() != gradient.rows() || _cache.cols() != gradient.cols()) {
				_cache = MatrixXd::Zero(gradient.rows(), gradient.cols());
			}

			_cache = _mu * _cache - learning_rate * gradient;
			_init = true;
			return _cache;
		}

	protected:
		bool _init;
		double _mu;
		MatrixXd _cache;
	};

	// Implementation of the Nesterov's Accelerated Momentum Gradient Descent update rule
	// Parameters:
	// - double mu: amount of momentum applied on each descent
	class NesterovMomentumGradientDescentUpdate {
	public:
		NesterovMomentumGradientDescentUpdate(double mu = 0.9) : _mu(mu), _init(false) {}

		void restart() { _init = false; }

		auto step(double learning_rate, MatrixXdRef gradient) {
			if (!_init || _cache.rows() != gradient.rows() || _cache.cols() != gradient.cols()) {
				_cache = MatrixXd::Zero(gradient.rows(), gradient.cols());
			}

			VectorXd velocity_prev = _cache;
			_cache = _mu * _cache - learning_rate * gradient;
			_init = true;
			return (-_mu * velocity_prev + (1 + _mu) * _cache).eval();
		}

	protected:
		bool _init;
		double _mu;
		MatrixXd _cache;
	};

	// Implementation of the Adagrad Gradient Descent update rule
	class AdagradGradientDescentUpdate {
	public:
		AdagradGradientDescentUpdate() : _init(false) {}

		void restart() { _init = false; }

		auto step(double learning_rate, MatrixXdRef gradient) {
			if (!_init || _cache.rows() != gradient.rows() || _cache.cols() != gradient.cols()) {
				_cache = MatrixXd::Zero(gradient.rows(), gradient.cols());
			}

			_cache += gradient.array().pow(2).matrix();
			_init = true;
			return (-learning_rate * (gradient.array() / (_cache.array() + 1e-8).sqrt()).matrix()).eval();
		}

	protected:
		bool _init;
		MatrixXd _cache;
	};

	// Implementation of the RMSProp Gradient Descent update rule
	// Parameters:
	// - double decay_rate: the decay rate of the moving average of squared gradients at each step
	class RMSPropGradientDescentUpdate {
	public:
		RMSPropGradientDescentUpdate(double decay_rate = 0.9) : _decay_rate(decay_rate), _init(false) {}

		void restart() { _init = false; }

		MatrixXd step(double learning_rate, MatrixXdRef gradient) {
			if (!_init || _cache.rows() != gradient.rows() || _cache.cols() != gradient.cols()) {
				_cache = MatrixXd::Zero(gradient.rows(), gradient.cols());
			}

			_cache = _decay_rate * _cache + (1 - _decay_rate) * gradient.array().pow(2).matrix();
			_init = true;
			return (-learning_rate * (gradient.array() / (_cache.array() + 1e-8).sqrt()).matrix()).eval();
		}
		
	protected:
		bool _init;
		double _decay_rate;
		MatrixXd _cache;
	};
}
}
}
#endif