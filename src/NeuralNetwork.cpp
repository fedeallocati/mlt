#include "NeuralNetwork.h"

NeuralNetwork::NeuralNetwork(unsigned int inputLayer, std::vector<unsigned int> hiddenLayers, unsigned int outputLayer)
{
	this->init(inputLayer, outputLayer, hiddenLayers);
}

NeuralNetwork::NeuralNetwork(unsigned int inputLayer, std::vector<unsigned int> hiddenLayers , unsigned int outputLayer, std::vector<Eigen::MatrixXd> trainedTheta)	
{
	this->init(inputLayer, outputLayer, hiddenLayers);
	this->trainedTheta = trainedTheta;
}

void NeuralNetwork::train(const Eigen::MatrixXd& trainingSet, const Eigen::VectorXi& labels, double lambda)
{
	unsigned int m = trainingSet.rows();

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

	this->lambda = lambda;

	unsigned int size = 0;

	for (int i = 0; i < this->trainedTheta.size(); i++)
	{
		size += this->trainedTheta[i].size();
	}
	
	dlib::matrix<double, 0, 1> nn_params;

	nn_params.set_size(size);

	unsigned int counter = 0;

	for (unsigned int i = 0; i < this->trainedTheta.size(); i++)
	{
		for(unsigned int j = 0; j < this->trainedTheta[i].size(); j++)
		{		
			nn_params(j + counter) = this->trainedTheta[i].data()[j];
		}

		counter += this->trainedTheta[i].size();
	}

	dlib::find_min(dlib::lbfgs_search_strategy(50),  // Use L-BFGS search algorithm
		dlib::objective_delta_stop_strategy(1e-7, 100), // Stop when the change in cost() is less than 1e-7
				 NeuralNetwork::costWrap(this, x, y), NeuralNetwork::gradientWrap(this, x, y), nn_params, -1);

	counter = 0;

	for(unsigned int i = 0; i < this->trainedTheta.size(); i++)
	{
		for(unsigned int j = 0; j < this->trainedTheta[i].size(); j++)
		{
			this->trainedTheta[i].data()[j] = nn_params(counter + j);
		}

		counter += this->trainedTheta[i].size();
	}
}

Eigen::VectorXi NeuralNetwork::predictMany(const Eigen::MatrixXd& features, Eigen::MatrixXd& confidences)
{
	Eigen::MatrixXd input(features.rows(), features.cols() + 1);
	input.block(0, 1, features.rows(), features.cols()) = features;
	input.col(0) = Eigen::VectorXd::Ones(features.rows());

	confidences = this->fowardProp(this->trainedTheta, input);
	Eigen::VectorXi prediction(confidences.cols());

	for(unsigned int i = 0; i < confidences.cols(); i++)
	{
		Eigen::MatrixXd::Index maxRow, maxCol;
		double max = confidences.col(i).maxCoeff(&maxRow, &maxCol);

		if (this->layers.back() == 1)
		{
			prediction(i) = round(max);
		}
		else
		{
			prediction(i) = maxRow;
		}
	}

	return prediction;
}

Eigen::VectorXi NeuralNetwork::predictMany(const Eigen::MatrixXd& features)
{
	Eigen::MatrixXd confidences;
	return this->predictMany(features, confidences);
}

unsigned int NeuralNetwork::predictOne(const Eigen::VectorXd& features, Eigen::VectorXd& confidence)
{
	Eigen::MatrixXd confidences = confidence;

	unsigned int pred = this->predictMany(features.transpose(), confidences)(0);
	confidence = confidences.col(0);

	return pred;
}

unsigned int NeuralNetwork::predictOne(const Eigen::VectorXd& features)
{
	Eigen::VectorXd confidence;
	return this->predictOne(features, confidence);
}

const std::vector<Eigen::MatrixXd> NeuralNetwork::getTheta() const
{
	return this->trainedTheta;
}

void NeuralNetwork::saveTheta(const char* file)
{
	std::ofstream f(file, std::ios::binary);

	for(unsigned int i = 0; i < this->trainedTheta.size(); i++)
	{
		Eigen::MatrixXd::Index rows, cols;
		rows = this->trainedTheta[i].rows();
		cols = this->trainedTheta[i].cols();

		f.write((char *)&rows, sizeof(rows));
		f.write((char *)&cols, sizeof(cols));
		f.write((char *)this->trainedTheta[i].data(), sizeof(Eigen::MatrixXd::Scalar) * rows * cols);
	}

	f.close();
}

void NeuralNetwork::loadTheta(const char* file)
{
	std::ifstream f(file, std::ios::binary);

	for(unsigned int i = 0; i < this->trainedTheta.size(); i++)
	{
		Eigen::MatrixXd::Index rows, cols;
	
		f.read((char *)&rows, sizeof(rows));
		f.read((char *)&cols, sizeof(cols));

		this->trainedTheta[i].resize(rows, cols);

		f.read((char *)this->trainedTheta[i].data(), sizeof(Eigen::MatrixXd::Scalar) * rows * cols);

		if (f.bad())
		{
			throw "Error reading matrix";
		}
	}

	f.close();
}

void NeuralNetwork::init(unsigned int inputLayer, unsigned int outputLayer, std::vector<unsigned int> hiddenLayers)
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

	this->lambda = 0;
}

void NeuralNetwork::initializeRandWeights()
{
	this->trainedTheta.clear();
	
	double epsilonInit = 0.12;

	for(unsigned int i = 1; i < this->layers.size(); i++)
	{
		this->trainedTheta.push_back(Eigen::MatrixXd::Random(this->layers[i], this->layers[i - 1] + 1) * epsilonInit);		
	}
}

Eigen::MatrixXd NeuralNetwork::fowardProp(const std::vector<Eigen::MatrixXd>& theta, const Eigen::MatrixXd& input)
{
	Eigen::MatrixXd previous = input.transpose();

	for(unsigned int i = 1; i < this->layers.size() - 1; i++)
	{
		Eigen::MatrixXd temp = Eigen::MatrixXd::Ones(this->layers[i] + 1, input.rows());
		temp.bottomRows(this->layers[i]) = (theta[i - 1] * previous).unaryExpr(std::ptr_fun(sigmoid));
		previous = temp;
	}
		
	return (theta.back() * previous).unaryExpr(std::ptr_fun(sigmoid));
}

Eigen::MatrixXd NeuralNetwork::fowardProp(const std::vector<Eigen::MatrixXd>& theta, const Eigen::MatrixXd& input, std::vector<Eigen::MatrixXd>& z, std::vector<Eigen::MatrixXd>& a)
{
	a.push_back(input.transpose());
	
	for(unsigned int i = 1; i < this->layers.size() - 1; i++)
	{
		z.push_back(theta[i - 1] * a.back());

		Eigen::MatrixXd temp = Eigen::MatrixXd::Ones(this->layers[i] + 1, input.rows());		
		temp.bottomRows(this->layers[i]) = (z.back()).unaryExpr(std::ptr_fun(sigmoid));
		a.push_back(temp);
	}
	
	a.push_back((theta.back() * a.back()).unaryExpr(std::ptr_fun(sigmoid)));
		
	return a.back();
}

std::vector<Eigen::MatrixXd> NeuralNetwork::backwardProp(const std::vector<Eigen::MatrixXd>& theta, const Eigen::MatrixXd& y, std::vector<Eigen::MatrixXd>& z, std::vector<Eigen::MatrixXd>& a)
{
	std::vector<Eigen::MatrixXd> grads;

	Eigen::MatrixXd previousDelta = a.back() - y;
	grads.push_back(previousDelta * a[a.size() - 2].transpose() / y.cols());

	for (unsigned int i = this->layers.size() - 2; i > 0 ; i--)
	{
		Eigen::MatrixXd temp = (theta[i].rightCols(this->layers[i]).transpose() * previousDelta).array() * z[i - 1].unaryExpr(std::ptr_fun(sigmoidGradient)).array();
		grads.push_back(temp * a[i - 1].transpose() / y.cols());
		previousDelta = temp;
	}

	for(int i = 0; i < grads.size(); i++)
	{		
		Eigen::MatrixXd temp = theta[grads.size() - 1 - i] * (this->lambda / y.cols());
		temp.col(0) = Eigen::VectorXd::Zero(temp.rows());		
		grads[i] += temp;
	}
	
	std::reverse(grads.begin(), grads.end());

	return grads;
}

double NeuralNetwork::cost(const std::vector<Eigen::MatrixXd>& theta, const Eigen::MatrixXd& x, const Eigen::MatrixXd& y)
{
	Eigen::MatrixXd output = this->fowardProp(theta, x);
		
	Eigen::MatrixXd ones = Eigen::MatrixXd::Ones(this->layers.back(), x.rows());
	
	double cost = ((-y).array() * output.array().log() - (ones - y).array() * (ones - output).array().log()).sum() / x.rows();

	double reg = 0;

	for(unsigned int i = 0; i < theta.size(); i++)
	{
		Eigen::MatrixXd temp = theta[i];
		temp.leftCols(1) = Eigen::VectorXd::Zero(theta[i].rows());
		reg += (temp.array().pow(2)).sum();
	}

	reg = reg * (this->lambda / (double)(2 * x.rows()));

	cost += reg;

	return cost;
}

std::vector<Eigen::MatrixXd> NeuralNetwork::gradient(const std::vector<Eigen::MatrixXd>& theta, const Eigen::MatrixXd& x, const Eigen::MatrixXd& y)
{
	std::vector<Eigen::MatrixXd> z;
	std::vector<Eigen::MatrixXd> a;
	
	this->fowardProp(theta, x, z, a);
	
	return this->backwardProp(theta, y, z, a);
}