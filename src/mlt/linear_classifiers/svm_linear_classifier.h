#ifndef SVM_LINEAR_CLASSIFIER_H
#define SVM_LINEAR_CLASSIFIER_H

#include <Eigen/Core>

#include "gradient_descent_trainable_linear_classifier.h"

namespace MLT
{
namespace LinearClassifiers
{	
	using namespace Eigen;

	class SvmLinearClassifier : public GradientDescentTrainableLinearClassifier
	{
	public:
		SvmLinearClassifier(size_t input, size_t output, double initial_epsilon, double lambda) :
			GradientDescentTrainableLinearClassifier(input, output, initial_epsilon), _lambda(lambda)
		{
		}

		double loss(const VectorXd& parameters, const MatrixXd& x, const MatrixXd& y) const
		{
			MatrixXd theta = MatrixXd::Map(parameters.data(), this->_output, this->_input + 1);
			
			MatrixXd scores = this->_score(theta, x);

			double loss = ((scores.rowwise() - (scores.array() * y.array()).colwise().sum().matrix()).array() + 1 - y.array()).max(0).sum();
			loss /= x.rows();
			loss += 0.5 * this->_lambda * (theta.array().pow(2)).sum();

			return loss;
		}

		VectorXd gradient(const VectorXd& parameters, const MatrixXd& x, const MatrixXd& y) const
		{
			MatrixXd theta = MatrixXd::Map(parameters.data(), this->_output, this->_input + 1);
			
			MatrixXd scores = this->_score(theta, x);

			MatrixXd marginMask = (((scores.rowwise() - (scores.array() * y.array()).colwise().sum().matrix()).array() + 1 - y.array()).max(0) > 0).cast<double>();
			marginMask = marginMask.array() + (y.array().rowwise() * -marginMask.colwise().sum().array());

			MatrixXd d_theta = marginMask * x;
			d_theta /= x.rows();
			d_theta += this->_lambda * theta;

			return VectorXd::Map(d_theta.data(), d_theta.size());
		}

	protected:
		double _lambda;
	};
}
}
#endif // SVM_LINEAR_CLASSIFIER_H