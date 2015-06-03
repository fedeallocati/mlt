#ifndef SVM_LINEAR_CLASSIFIER_H
#define SVM_LINEAR_CLASSIFIER_H

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
			
			MatrixXd scores = this->score(theta, x);			
			double loss = -(scores.array() * y.array()).colwise().sum().array().log().sum();
			loss /= x.rows();
			loss += 0.5 * this->_lambda * (theta.array().pow(2)).sum();

			return loss;
		}

		VectorXd gradient(const VectorXd& parameters, const MatrixXd& x, const MatrixXd& y) const
		{
			MatrixXd theta = MatrixXd::Map(parameters.data(), this->_output, this->_input + 1);
			
			MatrixXd scores = this->score(theta, x);
			MatrixXd dTheta = scores * x;
			dTheta -= y * x;
			dTheta /= x.rows();
			dTheta += this->_lambda * theta;

			return VectorXd::Map(dTheta.data(), dTheta.size());
		}		

	protected:
		double _lambda;
	};
}
}
#endif // SVM_LINEAR_CLASSIFIER_H