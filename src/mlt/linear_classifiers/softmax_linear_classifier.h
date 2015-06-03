#ifndef SOFTMAX_LINEAR_CLASSIFIER_H
#define SOFTMAX_LINEAR_CLASSIFIER_H

#include <Eigen/Core>

#include "gradient_descent_trainable_linear_classifier.h"

namespace MLT
{
namespace LinearClassifiers
{	
	using namespace Eigen;

	class SoftmaxLinearClassifier : public GradientDescentTrainableLinearClassifier
	{
	public:
		SoftmaxLinearClassifier(size_t input, size_t output, double initial_epsilon, double lambda) : 
			GradientDescentTrainableLinearClassifier(input, output, initial_epsilon), _lambda(lambda)
		{
		}

		double loss(const VectorXd& parameters, const MatrixXd& x, const MatrixXd& y) const
		{
			MatrixXd theta = MatrixXd::Map(parameters.data(), this->_output, this->_input + 1);
			
			Eigen::MatrixXd scores = this->_score(theta, x);
			scores.rowwise() -= scores.colwise().maxCoeff();
			scores = scores.array().exp();
			scores = scores.array().rowwise() / scores.colwise().sum().array();

			double loss = -(scores.array() * y.array()).colwise().sum().array().log().sum();
			loss /= x.rows();
			loss += 0.5 * this->_lambda * (theta.array().pow(2)).sum();

			return loss;
		}

		VectorXd gradient(const VectorXd& parameters, const MatrixXd& x, const MatrixXd& y) const
		{
			MatrixXd theta = MatrixXd::Map(parameters.data(), this->_output, this->_input + 1);
			
			MatrixXd scores = this->_score(theta, x);
			scores.rowwise() -= scores.colwise().maxCoeff();
			scores = scores.array().exp();
			scores = scores.array().rowwise() / scores.colwise().sum().array();

			MatrixXd d_theta = scores * x;
			d_theta -= y * x;
			d_theta /= x.rows();
			d_theta += this->_lambda * theta;

			return VectorXd::Map(d_theta.data(), d_theta.size());
		}

	protected:
		double _lambda;
	};
}
}
#endif // SOFTMAX_LINEAR_CLASSIFIER_H