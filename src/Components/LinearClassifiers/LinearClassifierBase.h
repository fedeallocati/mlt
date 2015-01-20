#ifndef LINEAR_CLASSSIFIER_BASE_H
#define LINEAR_CLASSSIFIER_BASE_H

#include <EigenOptimization/Optimization>

class LinearClassifierBase
{	
public:
	LinearClassifierBase(size_t input, size_t output);
	LinearClassifierBase(const Eigen::MatrixXd& theta);

	void train(const Eigen::MatrixXd& trainingSet, const Eigen::VectorXi& labels, Eigen::SearchStrategy& searchStrategy, Eigen::StopStrategy& stopStrategy, double lambda);

	Eigen::VectorXi predict(const Eigen::MatrixXd& features, Eigen::MatrixXd& confidences);
	Eigen::VectorXi predict(const Eigen::MatrixXd& features);

	const Eigen::MatrixXd& getTheta() const;
	void setTheta(const Eigen::MatrixXd& theta);

protected:
	virtual double lossInternal(const Eigen::MatrixXd& theta, const Eigen::MatrixXd& x, const Eigen::MatrixXd& y, double lambda) = 0;
	virtual Eigen::MatrixXd gradientInternal(const Eigen::MatrixXd& theta, const Eigen::MatrixXd& x, const Eigen::MatrixXd& y, double lambda) = 0;
	virtual Eigen::MatrixXd predictInternal(const Eigen::MatrixXd& theta, const Eigen::MatrixXd& x) = 0;

private:
	class Cost
	{	
	public:
		Cost(LinearClassifierBase* classifier, size_t& input, size_t& output, const Eigen::MatrixXd& x, const Eigen::MatrixXd& y, double lambda) : 
		   classifier(classifier), input(input), output(output), x(x), y(y), lambda(lambda)
		{
		}
	
		double operator()(const Eigen::VectorXd& params) const;

	private:
		LinearClassifierBase* classifier;
		size_t input;
		size_t output;
		const Eigen::MatrixXd& x, y;
		double lambda;
	};

	class Gradient
	{	
	public:
		Gradient(LinearClassifierBase* classifier, size_t& input, size_t& output, const Eigen::MatrixXd& x, const Eigen::MatrixXd& y, double lambda) : 
		  classifier(classifier), input(input), output(output), x(x), y(y), lambda(lambda)
		{
		}
	
		Eigen::VectorXd operator()(const Eigen::VectorXd& params) const;

	private:
		LinearClassifierBase* classifier;
		size_t input;
		size_t output;
		const Eigen::MatrixXd& x, y;
		double lambda;
	};		

	friend class Cost;
	friend class Gradient;

	size_t input;
	size_t output;
	Eigen::MatrixXd theta;
};

#endif // LINEAR_CLASSSIFIER_BASE_H