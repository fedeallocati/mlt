#include "LinearClassifierBase.h"

#include <iostream>

LinearClassifierBase::LinearClassifierBase(size_t input, size_t output)
{
	if (input < 1)
	{
		throw "Invalid input layer size. Must have at least 1 feature";
	}

	if (output < 2)
	{
		throw "Invalid output layer size. Must have at least 2 classes";
	}

	this->input = input + 1;
	this->output = output;
	this->theta = Eigen::MatrixXd::Random(this->output, this->input) * 0.001;
	//std::cout << this->theta.topRightCorner(5, 5) << std::endl << std::endl;
}

LinearClassifierBase::LinearClassifierBase(const Eigen::MatrixXd& theta)
{
	if (theta.cols() < 2)
	{
		throw "Invalid input size. Must have at least 1 feature";
	}

	if (theta.rows() < 2)
	{
		throw "Invalid output size. Must have at least 2 classes";
	}

	this->input = theta.cols();
	this->output = theta.rows();

	this->theta = theta;
}

void LinearClassifierBase::train(const Eigen::MatrixXd& trainingSet, const Eigen::VectorXi& labels, Eigen::SearchStrategy& searchStrategy, Eigen::StopStrategy& stopStrategy, double lambda)
{
	size_t m = trainingSet.rows();

	Eigen::MatrixXd x(m, trainingSet.cols() + 1);	
	x.block(0, 1, m, trainingSet.cols()) = trainingSet;
	x.col(0) = Eigen::VectorXd::Ones(m);
	
	Eigen::MatrixXd y = Eigen::MatrixXd::Zero(this->output, m);

	for(unsigned int i = 0; i < m; i++)
	{		
		y(labels(i), i) = 1;
	}
		
	size_t size = 0;		
	
	Eigen::VectorXd params = Eigen::VectorXd::Map(this->theta.data(), this->theta.size());

	//std::cout << params.topRows(25) << std::endl << std::endl;
	
	FindMin(searchStrategy, stopStrategy, Cost(this, this->input, this->output, x, y, lambda), 
		Gradient(this, this->input, this->output, x, y, lambda), params, -1);	

	for (size_t i = 0; i < this->theta.size(); i++)
	{
		this->theta.data()[i] = params(i);
	}
}

Eigen::VectorXi LinearClassifierBase::predict(const Eigen::MatrixXd& features, Eigen::MatrixXd& confidences)
{
	Eigen::MatrixXd input(features.rows(), features.cols() + 1);
	input.block(0, 1, features.rows(), features.cols()) = features;
	input.col(0) = Eigen::VectorXd::Ones(features.rows());

	confidences = this->predictInternal(this->theta, input);
		
	Eigen::VectorXi prediction(confidences.cols());

	for (size_t i = 0; i < confidences.cols(); i++)
	{
		Eigen::MatrixXd::Index maxRow, maxCol;
		double max = confidences.col(i).maxCoeff(&maxRow, &maxCol);
	
		prediction(i) = (int)maxRow;
	}

	return prediction;
}

Eigen::VectorXi LinearClassifierBase::predict(const Eigen::MatrixXd& features)
{
	Eigen::MatrixXd confidences;
	return this->predict(features, confidences);
}

const Eigen::MatrixXd& LinearClassifierBase::getTheta() const
{
	return this->theta;
}

void LinearClassifierBase::setTheta(const Eigen::MatrixXd& theta)
{
	this->theta = theta;
}

double LinearClassifierBase::Cost::operator()(const Eigen::VectorXd& params) const
{
	Eigen::MatrixXd theta = Eigen::MatrixXd::Map(params.data(), this->output, this->input);			
	
	return this->classifier->lossInternal(theta, this->x, this->y, this->lambda);
}

Eigen::VectorXd LinearClassifierBase::Gradient::operator()(const Eigen::VectorXd& params) const
{
	Eigen::MatrixXd theta = Eigen::MatrixXd::Map(params.data(), this->output, this->input);

	Eigen::MatrixXd gradient = this->classifier->gradientInternal(theta, this->x, this->y, this->lambda);
	
	return Eigen::VectorXd::Map(gradient.data(), gradient.size());
}