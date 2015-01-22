#ifndef SOFTMAX_LINEAR_CLASSIFIER_H
#define SOFTMAX_LINEAR_CLASSIFIER_H

#include "LinearClassifierBase.h"

class SoftmaxLinearClassifier : public LinearClassifierBase<SoftmaxLinearClassifier>
{	
public:
	friend class LinearClassifierBase<SoftmaxLinearClassifier>;

	SoftmaxLinearClassifier(size_t input, size_t output, double epsilonInit = 0.001, double lambda = 0) : 
		lambda(lambda), LinearClassifierBase(input, output, epsilonInit) { }

	SoftmaxLinearClassifier(const Eigen::MatrixXd& theta, double lambda = 0) : lambda(lambda), LinearClassifierBase(theta)	{}
	
private:
	double loss(const Eigen::MatrixXd& theta, const Eigen::MatrixXd& x, const Eigen::MatrixXd& y)
	{
		Eigen::MatrixXd scores = this->score(theta, x);
		double loss = -(scores.array() * y.array()).colwise().sum().array().log().sum();
		loss /= x.rows();
		loss += 0.5 * this->lambda * (theta.array().pow(2)).sum();

		return loss;
	}

	Eigen::MatrixXd gradient(const Eigen::MatrixXd& theta, const Eigen::MatrixXd& x, const Eigen::MatrixXd& y)
	{
		Eigen::MatrixXd scores = this->score(theta, x);
		Eigen::MatrixXd dTheta = scores * x;
		dTheta -= y * x;
		dTheta /= x.rows();
		dTheta += this->lambda * theta;

		return dTheta;
	}

	Eigen::MatrixXd predict(const Eigen::MatrixXd& theta, const Eigen::MatrixXd& x)
	{
		return this->score(theta, x);
	}

	Eigen::MatrixXd score(const Eigen::MatrixXd& theta, const Eigen::MatrixXd& x)
	{
		Eigen::MatrixXd scores = theta * x.transpose();
		scores.rowwise() -= scores.colwise().maxCoeff();
		scores = scores.array().exp();
		scores = scores.array().rowwise() / scores.colwise().sum().array();

		return scores;
	}

	double lambda;
};

#endif // SOFTMAX_LINEAR_CLASSIFIER_H