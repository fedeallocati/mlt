#ifndef FEEDFORWARD_NEURAL_NETWORK_H
#define FEEDFORWARD_NEURAL_NETWORK_H

#include <algorithm>
#include <vector>

#include <EigenOptimization/Optimization>

class FeedForwardNeuralNetwork
{
public:
	FeedForwardNeuralNetwork(size_t inputLayer, std::vector<size_t> hiddenLayers, size_t outputLayer);
	FeedForwardNeuralNetwork(size_t inputLayer, std::vector<size_t> hiddenLayers, size_t outputLayer, std::vector<Eigen::MatrixXd> trainedTheta);

	void train(const Eigen::MatrixXd& trainingSet, const Eigen::VectorXi& labels, Eigen::SearchStrategy& searchStrategy, Eigen::StopStrategy& stopStrategy, double lambda);

	Eigen::VectorXi predictMany(const Eigen::MatrixXd& features, Eigen::MatrixXd& confidences);
	Eigen::VectorXi predictMany(const Eigen::MatrixXd& features);
	unsigned int predictOne(const Eigen::VectorXd& features, Eigen::VectorXd& confidence);
	unsigned int predictOne(const Eigen::VectorXd& features);

	const std::vector<Eigen::MatrixXd>& getTheta() const;
	void setTheta(const std::vector<Eigen::MatrixXd>&);

	FeedForwardNeuralNetwork& debug(void (*debugCallback)(const std::vector<Eigen::MatrixXd>& theta))
    {
        this->isDebug = true;
		this->debugCallback = debugCallback;
        return *this;
    }

private:
	static Eigen::MatrixXd feedForward(const std::vector<Eigen::MatrixXd>& theta, const Eigen::MatrixXd& input);
	static Eigen::MatrixXd feedForward(const std::vector<Eigen::MatrixXd>& theta, const Eigen::MatrixXd& input, std::vector<Eigen::MatrixXd>& z, std::vector<Eigen::MatrixXd>& a);
	static std::vector<Eigen::MatrixXd> backPropagate(const std::vector<Eigen::MatrixXd>& theta, const Eigen::MatrixXd& y, std::vector<Eigen::MatrixXd>& z, std::vector<Eigen::MatrixXd>& a);

	class BackpropNNCost
	{	
	public:
		BackpropNNCost(std::vector<size_t>& layers, const Eigen::MatrixXd& x, const Eigen::MatrixXd& y, double lambda, bool debug, 
			void (*debugCallback)(const std::vector<Eigen::MatrixXd>& theta)) : layers(layers), x(x), y(y), lambda(lambda), debug(debug), debugCallback(debugCallback)
		{
		}
	
		double operator()(const Eigen::VectorXd& params) const;

	private:
		std::vector<size_t>& layers;
		const Eigen::MatrixXd& x, y;
		double lambda;
		bool debug;
		void (*debugCallback)(const std::vector<Eigen::MatrixXd>& theta);
	};

	class BackpropNNGradient
	{	
	public:
		BackpropNNGradient(std::vector<size_t>& layers, Eigen::MatrixXd& x, Eigen::MatrixXd& y, double lambda) : layers(layers), x(x), y(y), lambda(lambda)
		{
		}
	
		Eigen::VectorXd operator()(const Eigen::VectorXd& params) const;

	private:
		std::vector<size_t>& layers;
		Eigen::MatrixXd& x, y;
		double lambda;

	};

	void init(size_t inputLayer, size_t outputLayer, std::vector<size_t> hiddenLayers);
	void initializeRandWeights();
	
	std::vector<size_t> layers;
	std::vector<Eigen::MatrixXd> theta;
	bool isDebug;
	void (*debugCallback)(const std::vector<Eigen::MatrixXd>& theta);
};

double sigmoid(double z);

double sigmoidGradient(double z);

#endif // FEEDFORWARD_NEURAL_NETWORK_H