#ifndef LINEAR_CLASSSIFIER_BASE_H
#define LINEAR_CLASSSIFIER_BASE_H

#include <EigenOptimization/Optimization>

template <class Derived>
class LinearClassifierBase
{	
public:
	void train(const Eigen::MatrixXd& trainingSet, const Eigen::VectorXi& labels, Eigen::SearchStrategy& searchStrategy, Eigen::StopStrategy& stopStrategy)
	{
		size_t m = trainingSet.rows();

		Eigen::MatrixXd x(m, trainingSet.cols() + 1);
		x.block(0, 1, m, trainingSet.cols()) = trainingSet;
		x.col(0) = Eigen::VectorXd::Ones(m);

		Eigen::MatrixXd y = Eigen::MatrixXd::Zero(this->theta.rows(), m);

		for (unsigned int i = 0; i < m; i++)
		{
			y(labels(i), i) = 1;
		}

		size_t size = 0;

		Eigen::VectorXd params = Eigen::VectorXd::Map(this->theta.data(), this->theta.size());		

		FindMin(searchStrategy, stopStrategy, Cost(this, x, y), Gradient(this, x, y), params, -1);

		this->theta = Eigen::MatrixXd::Map(params.data(), this->theta.rows(), this->theta.cols());
	}

	Eigen::VectorXi predict(const Eigen::MatrixXd& features, Eigen::MatrixXd& confidences)
	{
		Eigen::MatrixXd input(features.rows(), features.cols() + 1);
		input.block(0, 1, features.rows(), features.cols()) = features;
		input.col(0) = Eigen::VectorXd::Ones(features.rows());

		confidences = static_cast<Derived*>(this)->predict(this->theta, input);

		Eigen::VectorXi prediction(confidences.cols());

		for (size_t i = 0; i < confidences.cols(); i++)
		{
			Eigen::MatrixXd::Index maxRow, maxCol;
			double max = confidences.col(i).maxCoeff(&maxRow, &maxCol);

			prediction(i) = (int)maxRow;
		}

		return prediction;
	}
	
	Eigen::VectorXi predict(const Eigen::MatrixXd& features)
	{
		Eigen::MatrixXd confidences;
		return this->predict(features, confidences);
	}

	const Eigen::MatrixXd& getTheta() const
	{
		return this->theta;
	}

	void setTheta(const Eigen::MatrixXd& theta)
	{
		if (theta.cols() < 2)
		{
			throw "Invalid input size. Must have at least 1 feature";
		}

		if (theta.rows() < 2)
		{
			throw "Invalid output size. Must have at least 2 classes";
		}

		this->theta = theta;
	}

protected:
	LinearClassifierBase(size_t input, size_t output, double epsilonInit = 0.001)
	{
		if (input < 1)
		{
			throw "Invalid input layer size. Must have at least 1 feature";
		}

		if (output < 2)
		{
			throw "Invalid output layer size. Must have at least 2 classes";
		}
				
		this->theta = Eigen::MatrixXd::Random(output, input + 1) * epsilonInit;
	}

	LinearClassifierBase(const Eigen::MatrixXd& theta)
	{
		this->setTheta(theta);
	}

private:
	class Cost
	{	
	public:
		Cost(LinearClassifierBase* classifier, const Eigen::MatrixXd& x, const Eigen::MatrixXd& y) : 
		   classifier(classifier), x(x), y(y)
		{
		}
	
		double operator()(const Eigen::VectorXd& params) const
		{
			Eigen::MatrixXd theta = Eigen::MatrixXd::Map(params.data(), this->classifier->theta.rows(), this->classifier->theta.cols());

			return static_cast<Derived*>(this->classifier)->loss(theta, this->x, this->y);
		}

	private:
		LinearClassifierBase* classifier;
		const Eigen::MatrixXd& x, y;
	};

	class Gradient
	{	
	public:
		Gradient(LinearClassifierBase* classifier, const Eigen::MatrixXd& x, const Eigen::MatrixXd& y) : 
		  classifier(classifier), x(x), y(y)
		{
		}
	
		Eigen::VectorXd operator()(const Eigen::VectorXd& params) const
		{
			Eigen::MatrixXd theta = Eigen::MatrixXd::Map(params.data(), this->classifier->theta.rows(), this->classifier->theta.cols());

			Eigen::MatrixXd gradient = static_cast<Derived*>(this->classifier)->gradient(theta, this->x, this->y);

			return Eigen::VectorXd::Map(gradient.data(), gradient.size());
		}

	private:
		LinearClassifierBase* classifier;
		const Eigen::MatrixXd& x, y;
		double lambda;
	};

	friend class Cost;
	friend class Gradient;
		
	Eigen::MatrixXd theta;
};

#endif // LINEAR_CLASSSIFIER_BASE_H