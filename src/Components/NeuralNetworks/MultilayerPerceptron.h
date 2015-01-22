#ifndef MULTILAYER_PERCEPTRON_H
#define MULTILAYER_PERCEPTRON_H

#include <algorithm>
#include <vector>
#include <EigenOptimization/Optimization>

#include "NeuralNetworkBase.h"
#include "../math.h"

class MultilayerPerceptron : public NeuralNetworkBase<MultilayerPerceptron>
{
public:
	MultilayerPerceptron(size_t inputLayer, std::vector<size_t> hiddenLayers, size_t outputLayer,
		double epsilonInit = 0.12, double lambda = 0)
		: lambda(lambda)
	{
		this->init(inputLayer, hiddenLayers, outputLayer, epsilonInit);
	}

	MultilayerPerceptron(std::vector<Eigen::MatrixXd>& theta, double lambda = 0) : lambda(lambda)
	{
		this->setTheta(theta);
	}

	void train(const Eigen::MatrixXd& trainingSet, const Eigen::VectorXi& labels,
		Eigen::SearchStrategy& searchStrategy, Eigen::StopStrategy& stopStrategy)
	{
		size_t m = trainingSet.rows();

		Eigen::MatrixXd x(m, trainingSet.cols() + 1);
		x.block(0, 1, m, trainingSet.cols()) = trainingSet;
		x.col(0) = Eigen::VectorXd::Ones(m);

		Eigen::MatrixXd y = Eigen::MatrixXd::Zero(this->theta.back().rows(), m);

		for (unsigned int i = 0; i < m; i++)
		{
			y(labels(i), i) = 1;
		}

		size_t size = 0;

		for (size_t i = 0; i < this->theta.size(); i++)
		{
			size += this->theta[i].size();
		}

		Eigen::VectorXd params(size);

		size_t counter = 0;

		for (size_t i = 0; i < this->theta.size(); i++)
		{
			params.block(counter, 0, this->theta[i].size(), 1) =
				Eigen::VectorXd::Map(this->theta[i].data(), this->theta[i].size());

			counter += this->theta[i].size();
		}

		FindMin(searchStrategy, stopStrategy, Cost(this, x, y), Gradient(this, x, y), params, -1);

		counter = 0;

		for (size_t i = 0; i < this->theta.size(); i++)
		{
			this->theta[i] = Eigen::MatrixXd::Map(params.data() + counter, this->theta[i].rows(), this->theta[i].cols());
			counter += this->theta[i].size();
		}
	}

	Eigen::VectorXi predict(const Eigen::MatrixXd& features, Eigen::MatrixXd& confidences)
	{
		Eigen::MatrixXd input(features.rows(), features.cols() + 1);
		input.block(0, 1, features.rows(), features.cols()) = features;
		input.col(0) = Eigen::VectorXd::Ones(features.rows());

		confidences = feedForward(this->theta, input);
		Eigen::VectorXi prediction(confidences.cols());

		for (size_t i = 0; i < confidences.cols(); i++)
		{
			Eigen::MatrixXd::Index maxRow, maxCol;
			double max = confidences.col(i).maxCoeff(&maxRow, &maxCol);

			prediction(i) = (int)maxRow;
		}

		return prediction;
	}

	const std::vector<Eigen::MatrixXd>& getTheta() const
	{
		return this->theta;
	}

	void setTheta(const std::vector<Eigen::MatrixXd>& theta)
	{
		if (theta.front().cols() < 2)
		{
			throw "Invalid input layer size. Must have at least 1 feature";
		}

		if (theta.back().rows() < 2)
		{
			throw "Invalid output layer size. Must have at least 2 classes";
		}

		this->theta = theta;
	}

private:
	static Eigen::MatrixXd feedForward(const std::vector<Eigen::MatrixXd>& theta, const Eigen::MatrixXd& input)
	{
		Eigen::MatrixXd previous = input.transpose();

		for (unsigned int i = 0; i < theta.size() - 1; i++)
		{
			Eigen::MatrixXd temp = Eigen::MatrixXd::Ones(theta[i].rows() + 1, input.rows());
			temp.bottomRows(theta[i].rows()) = (theta[i] * previous).unaryExpr(std::ptr_fun(sigmoid));
			previous = temp;
		}

		return (theta.back() * previous).unaryExpr(std::ptr_fun(sigmoid));
	}

	static Eigen::MatrixXd feedForward(const std::vector<Eigen::MatrixXd>& theta, const Eigen::MatrixXd& input, 
		std::vector<Eigen::MatrixXd>& z, std::vector<Eigen::MatrixXd>& a)
	{
		a.push_back(input.transpose());

		for (unsigned int i = 0; i < theta.size() - 1; i++)
		{
			Eigen::MatrixXd temp = Eigen::MatrixXd::Ones(theta[i].rows() + 1, input.rows());
			z.push_back(theta[i] * a.back());
			temp.bottomRows(theta[i].rows()) = (z.back()).unaryExpr(std::ptr_fun(sigmoid));
			a.push_back(temp);
		}

		a.push_back((theta.back() * a.back()).unaryExpr(std::ptr_fun(sigmoid)));

		return a.back();
	}

	static std::vector<Eigen::MatrixXd> backPropagate(const std::vector<Eigen::MatrixXd>& theta, const Eigen::MatrixXd& y,
		std::vector<Eigen::MatrixXd>& z, std::vector<Eigen::MatrixXd>& a)
	{
		std::vector<Eigen::MatrixXd> grads;

		Eigen::MatrixXd previousDelta = a.back() - y;
		grads.push_back(previousDelta * a[a.size() - 2].transpose() / (double)y.cols());

		for (size_t i = theta.size() - 1; i > 0; i--)
		{
			Eigen::MatrixXd temp = (theta[i].rightCols(theta[i].cols() - 1).transpose() * previousDelta).array() *
				z[i - 1].unaryExpr(std::ptr_fun(sigmoidGradient)).array();
			grads.push_back(temp * a[i - 1].transpose() / (double)y.cols());
			previousDelta = temp;
		}

		std::reverse(grads.begin(), grads.end());

		return grads;
	}

	double loss(const std::vector<Eigen::MatrixXd>& theta, const Eigen::MatrixXd& x, const Eigen::MatrixXd& y)
	{
		Eigen::MatrixXd output = feedForward(theta, x);

		Eigen::MatrixXd ones = Eigen::MatrixXd::Ones(theta.back().rows(), x.rows());

		double loss = ((-y).array() * output.array().log() - (ones - y).array() * (ones - output).array().log()).sum() / x.rows();

		double reg = 0;

		for (unsigned int i = 0; i < theta.size(); i++)
		{
			Eigen::MatrixXd temp = theta[i];
			temp.leftCols(1) = Eigen::VectorXd::Zero(theta[i].rows());
			reg += (temp.array().pow(2)).sum();
		}

		reg = reg * (this->lambda / (double)(2 * x.rows()));

		loss += reg;

		return loss;
	}

	std::vector<Eigen::MatrixXd> gradient(const std::vector<Eigen::MatrixXd>& theta, const Eigen::MatrixXd& x, const Eigen::MatrixXd& y)
	{
		std::vector<Eigen::MatrixXd> z, a;

		feedForward(theta, x, z, a);

		std::vector<Eigen::MatrixXd> dTheta = backPropagate(theta, y, z, a);

		for (size_t i = 0; i < dTheta.size(); i++)
		{
			Eigen::MatrixXd temp = theta[i] * (this->lambda / y.cols());
			temp.col(0) = Eigen::VectorXd::Zero(temp.rows());
			dTheta[i] += temp;
		}

		return dTheta;
	}

	class Cost
	{
	public:
		Cost(MultilayerPerceptron* mlp, const Eigen::MatrixXd& x, const Eigen::MatrixXd& y)
			: mlp(mlp), x(x), y(y) {}

		double operator()(const Eigen::VectorXd& params) const
		{
			std::vector<Eigen::MatrixXd> theta(this->mlp->theta.size());

			size_t counter = 0;

			for (size_t i = 0; i < this->mlp->theta.size(); i++)
			{
				theta[i] = Eigen::MatrixXd::Map(params.data() + counter, this->mlp->theta[i].rows(), this->mlp->theta[i].cols());
				counter += this->mlp->theta[i].size();
			}

			return this->mlp->loss(theta, x, y);
		}

	private:
		MultilayerPerceptron* mlp;
		const Eigen::MatrixXd& x, y;
	};

	class Gradient
	{
	public:
		Gradient(MultilayerPerceptron* mlp, Eigen::MatrixXd& x, Eigen::MatrixXd& y)
			: mlp(mlp), x(x), y(y) {}

		Eigen::VectorXd operator()(const Eigen::VectorXd& params) const
		{
			std::vector<Eigen::MatrixXd> theta(this->mlp->theta.size());

			size_t counter = 0;

			for (size_t i = 0; i < this->mlp->theta.size(); i++)
			{
				theta[i] = Eigen::MatrixXd::Map(params.data() + counter, this->mlp->theta[i].rows(), this->mlp->theta[i].cols());
				counter += this->mlp->theta[i].size();
			}
			
			std::vector<Eigen::MatrixXd> gradient = this->mlp->gradient(theta, x, y);

			Eigen::VectorXd gradientPlain(counter);

			counter = 0;

			for (size_t i = 0; i < gradient.size(); i++)
			{
				gradientPlain.block(counter, 0, gradient[i].size(), 1) =
					Eigen::VectorXd::Map(gradient[i].data(), gradient[i].size());

				counter += gradient[i].size();
			}

			return gradientPlain;
		}

	private:
		MultilayerPerceptron* mlp;
		const Eigen::MatrixXd& x, y;
	};

	friend class Cost;
	friend class Gradient;

	void init(size_t inputLayer, std::vector<size_t> hiddenLayers, size_t outputLayer, double epsilonInit)
	{
		if (inputLayer < 1)
		{
			throw "Invalid input layer size. Must have at least 1 feature";
		}

		if (outputLayer < 2)
		{
			throw "Invalid output layer size. Must have at least 2 classes";
		}

		this->theta.clear();

		size_t previousLayer = inputLayer;

		for (unsigned int i = 0; i < hiddenLayers.size(); i++)
		{
			this->theta.push_back(Eigen::MatrixXd::Random(hiddenLayers[i], previousLayer + 1) * epsilonInit);

			previousLayer = hiddenLayers[i];
		}

		this->theta.push_back(Eigen::MatrixXd::Random(outputLayer, previousLayer + 1) * epsilonInit);
	}

	double lambda;
	std::vector<Eigen::MatrixXd> theta;
};

#endif // MULTILAYER_PERCEPTRON_H