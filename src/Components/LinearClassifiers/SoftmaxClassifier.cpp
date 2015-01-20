#include "SoftmaxClassifier.h"

#include <iostream>

SoftmaxClassifier::SoftmaxClassifier(size_t input, size_t output) : LinearClassifierBase(input, output)
{
}

SoftmaxClassifier::SoftmaxClassifier(const Eigen::MatrixXd& theta) : LinearClassifierBase(theta)
{
}

double SoftmaxClassifier::lossInternal(const Eigen::MatrixXd& theta, const Eigen::MatrixXd& x, const Eigen::MatrixXd& y, double lambda)
{	
	Eigen::MatrixXd scores = this->score(theta, x);	
	double loss = -(scores.array() * y.array()).colwise().sum().array().log().sum();	
	loss /= x.rows();
	loss += 0.5 * lambda * (theta.array().pow(2)).sum();

	return loss;
}

Eigen::MatrixXd SoftmaxClassifier::gradientInternal(const Eigen::MatrixXd& theta, const Eigen::MatrixXd& x, const Eigen::MatrixXd& y, double lambda)
{
	Eigen::MatrixXd scores = this->score(theta, x);
	Eigen::MatrixXd dTheta = scores * x;
	dTheta -= y * x;
	dTheta /= x.rows();
	dTheta += lambda * theta;

	return dTheta;
}

Eigen::MatrixXd SoftmaxClassifier::predictInternal(const Eigen::MatrixXd& theta, const Eigen::MatrixXd& x)
{
	return this->score(theta, x);
}

Eigen::MatrixXd SoftmaxClassifier::score(const Eigen::MatrixXd& theta, const Eigen::MatrixXd& x)
{	
	Eigen::MatrixXd scores = theta * x.transpose();
	scores.rowwise() -= scores.colwise().maxCoeff();
	scores = scores.array().exp();
	scores = scores.array().rowwise() / scores.colwise().sum().array();

	return scores;
}