#ifndef MULTILAYER_PERCEPTRON_CLASSIFIER_H
#define MULTILAYER_PERCEPTRON_CLASSIFIER_H

#include <vector>

#include "neural_network_classifier.h"
#include "../base/igradient_descent_trainable.h"
#include "../math.h"

namespace MLT
{
namespace NeuralNetworks
{	
	using namespace Eigen;
	using namespace Base;

	class MultilayerPerceptronClassifier : public NeuralNetworkClassifier, public IGradientDescentTrainable
	{
	public:
		MultilayerPerceptronClassifier(size_t input, std::vector<size_t> hidden_layers, size_t output,
			double epsilonInit = 0.12, double lambda = 0)
			: NeuralNetworkClassifier(input, output), _lambda(lambda)
		{
			assert(input > 0);
			assert(output > 1);

			this->_theta.clear();

			size_t previousLayer = input;

			for (unsigned int i = 0; i < hidden_layers.size(); i++)
			{
				assert(hidden_layers[i] > 0);
				this->_theta.push_back(MatrixXd::Random(hidden_layers[i], previousLayer + 1) * epsilonInit);

				previousLayer = hidden_layers[i];
			}

			this->_theta.push_back(MatrixXd::Random(output, previousLayer + 1) * epsilonInit);
		}

		double loss(const VectorXd& parameters, const MatrixXd& x, const MatrixXd& y) const
		{
			std::vector<MatrixXd> theta(this->_theta.size());
			size_t counter = 0;
			for (size_t i = 0; i < this->_theta.size(); i++)
			{
				theta[i] = MatrixXd::Map(parameters.data() + counter, this->_theta[i].rows(), this->_theta[i].cols());
				counter += this->_theta[i].size();
			}

			MatrixXd output = this->_feed_forward(theta, x);

			MatrixXd ones = MatrixXd::Ones(theta.back().rows(), x.rows());
			double loss = ((-y).array() * output.array().log() - (ones - y).array() * (ones - output).array().log()).sum() / x.rows();
			double reg = 0;

			for (unsigned int i = 0; i < theta.size(); i++)
			{	
				reg += theta[i].rightCols(theta[i].cols() - 1).array().pow(2).sum();
			}

			reg *= (this->_lambda / (double)(2 * x.rows()));

			loss += reg;

			return loss;
		}

		VectorXd gradient(const VectorXd& parameters, const MatrixXd& x, const MatrixXd& y) const
		{
			std::vector<MatrixXd> theta(this->_theta.size());
			size_t counter = 0;
			for (size_t i = 0; i < this->_theta.size(); i++)
			{
				theta[i] = MatrixXd::Map(parameters.data() + counter, this->_theta[i].rows(), this->_theta[i].cols());
				counter += this->_theta[i].size();
			}

			std::vector<MatrixXd> z, a;
			this->_feed_forward(theta, x, z, a);
			std::vector<MatrixXd> d_theta = this->_back_propagate(theta, y, z, a);

			for (size_t i = 0; i < d_theta.size(); i++)
			{
				d_theta[i].rightCols(theta[i].cols() - 1) += (theta[i] * (this->_lambda / y.cols())).rightCols(theta[i].cols() - 1);
			}

			VectorXd gradient_plain(counter);
			counter = 0;
			for (size_t i = 0; i < d_theta.size(); i++)
			{
				gradient_plain.block(counter, 0, d_theta[i].size(), 1) = VectorXd::Map(d_theta[i].data(), d_theta[i].size());

				counter += d_theta[i].size();
			}

			return gradient_plain;
		}

		inline const VectorXd parameters() const
		{
			size_t counter = 0;
			for (size_t i = 0; i < this->_theta.size(); i++)
			{
				counter += this->_theta[i].size();
			}

			VectorXd theta_plain(counter);
			counter = 0;
			for (size_t i = 0; i < this->_theta.size(); i++)
			{
				theta_plain.block(counter, 0, this->_theta[i].size(), 1) = VectorXd::Map(this->_theta[i].data(), this->_theta[i].size());

				counter += this->_theta[i].size();
			}

			return theta_plain;
		}

		inline void set_parameters(const VectorXd& parameters)
		{
			size_t counter = 0;
			for (size_t i = 0; i < this->_theta.size(); i++)
			{
				counter += this->_theta[i].size();
			}

			assert(parameters.size() == counter);

			counter = 0;
			for (size_t i = 0; i < this->_theta.size(); i++)
			{
				this->_theta[i] = MatrixXd::Map(parameters.data() + counter, this->_theta[i].rows(), this->_theta[i].cols());
				counter += this->_theta[i].size();
			}			
		}

		inline bool add_intercept() const { return true; }

		inline const std::vector<MatrixXd>& theta() const
		{
			return this->_theta;
		}

		inline void set_theta(const std::vector<MatrixXd>& theta)
		{			
			assert(theta.front().cols() == this->_input + 1);
			assert(theta.back().rows() == this->_output);

			this->_theta = theta;
		}

	protected:
		MatrixXd _score(const MatrixXd& input) const
		{
			return this->_feed_forward(this->_theta, input);
		}

		MatrixXd _feed_forward(const std::vector<MatrixXd>& theta, const MatrixXd& input) const
		{
			MatrixXd previous = input.transpose();

			for (unsigned int i = 0; i < theta.size() - 1; i++)
			{
				MatrixXd temp = MatrixXd::Ones(theta[i].rows() + 1, input.rows());
				temp.bottomRows(theta[i].rows()) = (theta[i] * previous).unaryExpr(std::ptr_fun(sigmoid));
				previous = temp;
			}

			return (theta.back() * previous).unaryExpr(std::ptr_fun(sigmoid));
		}

		MatrixXd _feed_forward(const std::vector<MatrixXd>& theta, const MatrixXd& input, 
			std::vector<MatrixXd>& z, std::vector<MatrixXd>& a) const
		{
			a.push_back(input.transpose());

			for (unsigned int i = 0; i < theta.size() - 1; i++)
			{
				MatrixXd temp = MatrixXd::Ones(theta[i].rows() + 1, input.rows());
				z.push_back(theta[i] * a.back());
				temp.bottomRows(theta[i].rows()) = (z.back()).unaryExpr(std::ptr_fun(sigmoid));
				a.push_back(temp);
			}

			a.push_back((theta.back() * a.back()).unaryExpr(std::ptr_fun(sigmoid)));

			return a.back();
		}

		std::vector<MatrixXd> _back_propagate(const std::vector<MatrixXd>& theta, const MatrixXd& y,
			std::vector<MatrixXd>& z, std::vector<MatrixXd>& a) const
		{
			std::vector<MatrixXd> grads;

			MatrixXd previousDelta = a.back() - y;
			grads.push_back(previousDelta * a[a.size() - 2].transpose() / (double)y.cols());

			for (size_t i = theta.size() - 1; i > 0; i--)
			{
				MatrixXd temp = (theta[i].rightCols(theta[i].cols() - 1).transpose() * previousDelta).array() *
					z[i - 1].unaryExpr(std::ptr_fun(sigmoidGradient)).array();
				grads.push_back(temp * a[i - 1].transpose() / (double)y.cols());
				previousDelta = temp;
			}

			std::reverse(grads.begin(), grads.end());

			return grads;
		}

		std::vector<MatrixXd> _theta;
		double _lambda;
	};
}
}
#endif // MULTILAYER_PERCEPTRON_CLASSIFIER_H