#include "FeedForwardNeuralNetwork.h"

FeedForwardNeuralNetwork::FeedForwardNeuralNetwork(size_t inputLayer, std::vector<size_t> hiddenLayers, size_t outputLayer)
{
	this->init(inputLayer, outputLayer, hiddenLayers);
}

FeedForwardNeuralNetwork::FeedForwardNeuralNetwork(size_t inputLayer, std::vector<size_t> hiddenLayers , size_t outputLayer, std::vector<Eigen::MatrixXd> trainedTheta)	
{
	this->init(inputLayer, outputLayer, hiddenLayers);
	this->trainedTheta = trainedTheta;
}

void FeedForwardNeuralNetwork::train(const Eigen::MatrixXd& trainingSet, const Eigen::VectorXi& labels, double lambda)
{
	size_t m = trainingSet.rows();

	Eigen::MatrixXd x(m, trainingSet.cols() + 1);	
	x.block(0, 1, m, trainingSet.cols()) = trainingSet;
	x.col(0) = Eigen::VectorXd::Ones(m);
	
	Eigen::MatrixXd y = Eigen::MatrixXd::Zero(this->layers.back(), m);

	if (this->layers.back() == 1)
	{
		for(unsigned int i = 0; i < m; i++)
		{		
			y(0, i) = labels(i);
		}
	}
	else
	{
		for(unsigned int i = 0; i < m; i++)
		{		
			y(labels(i), i) = 1;
		}
	}
	
	size_t size = 0;

	for (int i = 0; i < this->trainedTheta.size(); i++)
	{
		size += this->trainedTheta[i].size();
	}	

	Eigen::VectorXd params(size);

	size_t counter = 0;

	for (unsigned int i = 0; i < this->trainedTheta.size(); i++)
	{
		for(unsigned int j = 0; j < this->trainedTheta[i].size(); j++)
		{		
			params(j + counter) = this->trainedTheta[i].data()[j];
		}

		counter += this->trainedTheta[i].size();
	}

	FindMin(Eigen::LBFGS(50), Eigen::ObjectiveDelta(1e-7, 100), BackpropNNCost(this->layers, x, y, lambda), BackpropNNGradient(this->layers, x, y, lambda), params, -1);

	counter = 0;

	for(unsigned int i = 0; i < this->trainedTheta.size(); i++)
	{
		for(unsigned int j = 0; j < this->trainedTheta[i].size(); j++)
		{
			this->trainedTheta[i].data()[j] = params(counter + j);
		}

		counter += this->trainedTheta[i].size();
	}
}

Eigen::VectorXi FeedForwardNeuralNetwork::predictMany(const Eigen::MatrixXd& features, Eigen::MatrixXd& confidences)
{
	Eigen::MatrixXd input(features.rows(), features.cols() + 1);
	input.block(0, 1, features.rows(), features.cols()) = features;
	input.col(0) = Eigen::VectorXd::Ones(features.rows());

	confidences = feedForward(this->trainedTheta, input);
	Eigen::VectorXi prediction(confidences.cols());

	for(unsigned int i = 0; i < confidences.cols(); i++)
	{
		Eigen::MatrixXd::Index maxRow, maxCol;
		double max = confidences.col(i).maxCoeff(&maxRow, &maxCol);

		if (this->layers.back() == 1)
		{
			prediction(i) = max >= 0.5;
		}
		else
		{
			prediction(i) = (int)maxRow;
		}
	}

	return prediction;
}

Eigen::VectorXi FeedForwardNeuralNetwork::predictMany(const Eigen::MatrixXd& features)
{
	Eigen::MatrixXd confidences;
	return this->predictMany(features, confidences);
}

unsigned int FeedForwardNeuralNetwork::predictOne(const Eigen::VectorXd& features, Eigen::VectorXd& confidence)
{
	Eigen::MatrixXd confidences = confidence;

	unsigned int pred = this->predictMany(features.transpose(), confidences)(0);
	confidence = confidences.col(0);

	return pred;
}

unsigned int FeedForwardNeuralNetwork::predictOne(const Eigen::VectorXd& features)
{
	Eigen::VectorXd confidence;
	return this->predictOne(features, confidence);
}

const std::vector<Eigen::MatrixXd>& FeedForwardNeuralNetwork::getTheta() const
{
	return this->trainedTheta;
}

Eigen::MatrixXd FeedForwardNeuralNetwork::feedForward(const std::vector<Eigen::MatrixXd>& theta, const Eigen::MatrixXd& input)
{
	Eigen::MatrixXd previous = input.transpose();

	for(unsigned int i = 0; i < theta.size() - 1; i++)
	{
		Eigen::MatrixXd temp = Eigen::MatrixXd::Ones(theta[i].rows() + 1, input.rows());		
		temp.bottomRows(theta[i].rows()) = (theta[i] * previous).unaryExpr(std::ptr_fun(sigmoid));
		previous = temp;
	}
		
	return (theta.back() * previous).unaryExpr(std::ptr_fun(sigmoid));
}

Eigen::MatrixXd FeedForwardNeuralNetwork::feedForward(const std::vector<Eigen::MatrixXd>& theta, const Eigen::MatrixXd& input, std::vector<Eigen::MatrixXd>& z, std::vector<Eigen::MatrixXd>& a)
{
	a.push_back(input.transpose());
	
	for(unsigned int i = 0; i < theta.size() - 1; i++)
	{
		Eigen::MatrixXd temp = Eigen::MatrixXd::Ones(theta[i].rows() + 1, input.rows());
		z.push_back(theta[i] * a.back());
		temp.bottomRows(theta[i].rows()) = (z.back()).unaryExpr(std::ptr_fun(sigmoid));
		a.push_back(temp);
	}	
	
	a.push_back((theta.back() * a.back()).unaryExpr(std::ptr_fun(sigmoid)));
		
	return a.back();
}

std::vector<Eigen::MatrixXd> FeedForwardNeuralNetwork::backPropagate(const std::vector<Eigen::MatrixXd>& theta, const Eigen::MatrixXd& y, std::vector<Eigen::MatrixXd>& z, std::vector<Eigen::MatrixXd>& a)
{
	std::vector<Eigen::MatrixXd> grads;

	Eigen::MatrixXd previousDelta = a.back() - y;
	grads.push_back(previousDelta * a[a.size() - 2].transpose() / (double)y.cols());

	for (size_t i = theta.size() - 1; i > 0 ; i--)
	{
		Eigen::MatrixXd temp = (theta[i].rightCols(theta[i].cols() - 1).transpose() * previousDelta).array() * z[i - 1].unaryExpr(std::ptr_fun(sigmoidGradient)).array();
		grads.push_back(temp * a[i - 1].transpose() / (double)y.cols());
		previousDelta = temp;
	}

	std::reverse(grads.begin(), grads.end());

	return grads;
}

double FeedForwardNeuralNetwork::BackpropNNCost::operator()(const Eigen::VectorXd& params) const
{
	std::vector<Eigen::MatrixXd> theta;
	size_t counter = 0;
						
	for(size_t i = 1; i < this->layers.size(); i++)
	{
		Eigen::MatrixXd curr(this->layers[i], this->layers[i - 1] + 1);

		for(unsigned int j = 0; j < curr.size(); j++)
		{
			curr.data()[j] = params(counter + j);
		}

		counter += curr.size();

		theta.push_back(curr);				
	}

	Eigen::MatrixXd output = feedForward(theta, x);
		
	Eigen::MatrixXd ones = Eigen::MatrixXd::Ones(theta.back().rows(), x.rows());
	
	double cost = ((-y).array() * output.array().log() - (ones - y).array() * (ones - output).array().log()).sum() / x.rows();

	double reg = 0;

	for(unsigned int i = 0; i < theta.size(); i++)
	{
		Eigen::MatrixXd temp = theta[i];
		temp.leftCols(1) = Eigen::VectorXd::Zero(theta[i].rows());
		reg += (temp.array().pow(2)).sum();
	}

	reg = reg * (lambda / (double)(2 * x.rows()));

	cost += reg;

	return cost;
}

Eigen::VectorXd FeedForwardNeuralNetwork::BackpropNNGradient::operator()(const Eigen::VectorXd& params) const
{
	std::vector<Eigen::MatrixXd> theta;
	size_t counter = 0;
						
	for(size_t i = 1; i < this->layers.size(); i++)
	{
		Eigen::MatrixXd curr(this->layers[i], this->layers[i - 1] + 1);

		for(unsigned int j = 0; j < curr.size(); j++)
		{
			curr.data()[j] = params(counter + j);
		}

		counter += curr.size();

		theta.push_back(curr);				
	}

	std::vector<Eigen::MatrixXd> z, a;
		
	feedForward(theta, x, z, a);
	
	std::vector<Eigen::MatrixXd> grads = backPropagate(theta, y, z, a);

	for(int i = 0; i < grads.size(); i++)
	{		
		Eigen::MatrixXd temp = theta[i] * (this->lambda / y.cols());
		temp.col(0) = Eigen::VectorXd::Zero(temp.rows());		
		grads[i] += temp;
	}		

	size_t size = 0;

	for(size_t i = 0; i < grads.size(); i++)
	{
		size += grads[i].size();
	}
		
	Eigen::VectorXd gradsPlain;

	gradsPlain.resize(size);

	counter = 0;

	for(size_t i = 0; i < grads.size() ; i++)
	{
		for(unsigned int j = 0; j < grads[i].size(); j++)
		{		
			gradsPlain(j + counter) = grads[i].data()[j];
		}

		counter += grads[i].size();
	}

	return gradsPlain;
}

void FeedForwardNeuralNetwork::init(size_t inputLayer, size_t outputLayer, std::vector<size_t> hiddenLayers)
{
	if (inputLayer == 0)
	{
		throw "Invalid input layer size. Must have at leaste 1 node";
	}

	if (outputLayer == 0)
	{
		throw "Invalid output layer size. Must have at leaste 1 node";
	}

	this->layers.push_back(inputLayer);
	this->layers.insert(this->layers.end(), hiddenLayers.begin(), hiddenLayers.end());
	this->layers.push_back(outputLayer);

	this->initializeRandWeights();
}

void FeedForwardNeuralNetwork::initializeRandWeights()
{
	this->trainedTheta.clear();
	
	double epsilonInit = 0.12;

	for(unsigned int i = 1; i < this->layers.size(); i++)
	{
		this->trainedTheta.push_back(Eigen::MatrixXd::Random(this->layers[i], this->layers[i - 1] + 1) * epsilonInit);		
	}
}