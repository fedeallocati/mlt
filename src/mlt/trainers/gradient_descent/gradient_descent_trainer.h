#ifndef GRADIENT_DESCENT_TRAINER_H
#define GRADIENT_DESCENT_TRAINER_H

#include <iostream>
#include "../../base/igradient_descent_trainable.h"
#include <EigenOptimization/Optimization>

namespace MLT
{
namespace Trainers
{
namespace GradientDescent
{
	using namespace Eigen;
	using namespace Base;

	class GradientDescentTrainer
	{
	public:		
		void train(IGradientDescentTrainable& trainee, const MatrixXd& training_data, const MatrixXd& training_target)
		{
			assert(training_data.rows() == training_target.cols());
			size_t m = training_data.rows();

			MatrixXd x;

			if (trainee.add_intercept())
			{
				x = MatrixXd(m, training_data.cols() + 1);
				x.block(0, 1, m, training_data.cols()) = training_data;
				x.col(0) = VectorXd::Ones(m);
			}
			else
			{
				x = training_data;
			}

			VectorXd params = trainee.parameters();

			LBFGS searchStrategy(50);
			ObjectiveDelta stopStrategy(1e-7, 250);

			auto loss = [&](const Eigen::VectorXd& parameters) { return trainee.loss(parameters, x, training_target); };
			auto gradient = [&](const Eigen::VectorXd& parameters) { return trainee.gradient(parameters, x, training_target); };

			FindMin(searchStrategy, stopStrategy, loss, gradient, params, -1);

			trainee.set_parameters(params);
		}
	};
}
}
}
#endif // GRADIENT_DESCENT_TRAINABLE_H