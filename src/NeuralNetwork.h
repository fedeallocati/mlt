#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include<vector>
#include<Eigen/Eigen>
#include<dlib/matrix.h>
#include "Utils.h"
#include <algorithm>
#include <dlib/optimization.h>
#include <fstream>
#include <math.h>

class NeuralNetwork
{
public:
	NeuralNetwork(unsigned int inputLayer, std::vector<unsigned int> hiddenLayers, unsigned int outputLayer);

	NeuralNetwork(unsigned int inputLayer, std::vector<unsigned int> hiddenLayers, unsigned int outputLayer, std::vector<Eigen::MatrixXd> trainedTheta);

	void train(const Eigen::MatrixXd& trainingSet, const Eigen::VectorXi& labels, double lambda = 0);

	Eigen::VectorXi predictMany(const Eigen::MatrixXd& features, Eigen::MatrixXd& confidences);
	Eigen::VectorXi predictMany(const Eigen::MatrixXd& features);
	unsigned int predictOne(const Eigen::VectorXd& features, Eigen::VectorXd& confidence);
	unsigned int predictOne(const Eigen::VectorXd& features);

	const std::vector<Eigen::MatrixXd> getTheta() const;
	void saveTheta(const char* file);
	void loadTheta(const char* file);

private:
	void init(unsigned int inputLayer, unsigned int outputLayer, std::vector<unsigned int> hiddenLayers);
	void initializeRandWeights();
	Eigen::MatrixXd fowardProp(const std::vector<Eigen::MatrixXd>& theta, const Eigen::MatrixXd& input);
	Eigen::MatrixXd fowardProp(const std::vector<Eigen::MatrixXd>& theta, const Eigen::MatrixXd& input, std::vector<Eigen::MatrixXd>& z, std::vector<Eigen::MatrixXd>& a);
	std::vector<Eigen::MatrixXd> backwardProp(const std::vector<Eigen::MatrixXd>& theta, const Eigen::MatrixXd& y, std::vector<Eigen::MatrixXd>& z, std::vector<Eigen::MatrixXd>& a);

	double cost(const std::vector<Eigen::MatrixXd>& theta, const Eigen::MatrixXd& x, const Eigen::MatrixXd& y);
	std::vector<Eigen::MatrixXd> gradient(const std::vector<Eigen::MatrixXd>& theta, const Eigen::MatrixXd& x, const Eigen::MatrixXd& y);

	class costWrap
	{
		friend class NeuralNetwork;
	public:		
		double operator() (const dlib::matrix<double, 0, 1>& nn_params) const
		{		
			std::vector<Eigen::MatrixXd> theta;
			unsigned int counter = 0;
						
			for(unsigned int i = 1; i < this->n->layers.size(); i++)
			{
				Eigen::MatrixXd curr(this->n->layers[i], this->n->layers[i - 1] + 1);

				for(unsigned int j = 0; j < curr.size(); j++)
				{
					curr.data()[j] = nn_params(counter + j);
				}

				counter += curr.size();

				theta.push_back(curr);				
			}

			return this->n->cost(theta, this->x, this->y);
		}

	private:
		costWrap(NeuralNetwork* n, Eigen::MatrixXd& x, Eigen::MatrixXd& y) : n(n), x(x), y(y)
		{			
		}
		
		NeuralNetwork* n;
		Eigen::MatrixXd& x, y;

	};

	class gradientWrap
	{
		friend class NeuralNetwork;
	public:		
		dlib::matrix<double, 0, 1> operator() (const dlib::matrix<double, 0, 1>& nn_params) const
		{
			std::vector<Eigen::MatrixXd> theta;
			unsigned int counter = 0;
						
			for(unsigned int i = 1; i < this->n->layers.size(); i++)
			{
				Eigen::MatrixXd curr(this->n->layers[i], this->n->layers[i - 1] + 1);

				for(unsigned int j = 0; j < curr.size(); j++)
				{
					curr.data()[j] = nn_params(counter + j);
				}

				counter += curr.size();

				theta.push_back(curr);				
			}

			std::vector<Eigen::MatrixXd> grads = this->n->gradient(theta, this->x, this->y);

			unsigned int size = 0;

			for (int i = 0; i < grads.size(); i++)
			{
				size += grads[i].size();
			}
		
			dlib::matrix<double, 0, 1> grad;

			grad.set_size(size);

			counter = 0;

			for (unsigned int i = 0; i < grads.size() ; i++)
			{
				for(unsigned int j = 0; j < grads[i].size(); j++)
				{		
					grad(j + counter) = grads[i].data()[j];
				}

				counter += grads[i].size();
			}

			return grad;
		}

	private:
		gradientWrap(NeuralNetwork* n, Eigen::MatrixXd& x, Eigen::MatrixXd& y) : n(n), x(x), y(y)
		{
		}
		
		NeuralNetwork* n;
		Eigen::MatrixXd& x, y;
	};

	std::vector<unsigned int> layers;
	std::vector<Eigen::MatrixXd> trainedTheta;
		
	double lambda;
};

#endif // NEURALNETWORK_H