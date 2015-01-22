#ifndef NEURAL_NETWORK_BASE_H
#define NEURAL_NETWORK_BASE_H

template <class Derived>
class NeuralNetworkBase
{
public:	
	void train(const Eigen::MatrixXd& trainingSet, const Eigen::VectorXi& labels, Eigen::SearchStrategy& searchStrategy, Eigen::StopStrategy& stopStrategy)
	{
		return static_cast<Derived*>(this)->train(trainingSet, labels, searchStrategy, stopStrategy);
	}

	Eigen::VectorXi predict(const Eigen::MatrixXd& features, Eigen::MatrixXd& confidences)
	{
		return static_cast<Derived*>(this)->predict(features, confidences);
	}

	Eigen::VectorXi predict(const Eigen::MatrixXd& features)
	{
		Eigen::MatrixXd confidences;
		return this->predict(features, confidences);
	}

protected:
	NeuralNetworkBase() {}
};

#endif // NEURAL_NETWORK_BASE_H