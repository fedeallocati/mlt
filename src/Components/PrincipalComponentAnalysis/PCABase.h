#ifndef PCA_BASE_H
#define PCA_BASE_H

template <class Derived>
class PCABase
{
public:
	void train(const Eigen::MatrixXd& trainingSet, size_t k = 0)
	{
		static_cast<Derived*>(this)->train(trainingSet, k);
	}
		
	Eigen::MatrixXd transform(const Eigen::MatrixXd& x)
	{
		return static_cast<Derived*>(this)->transform(x);
	}

	Eigen::MatrixXd recover(const Eigen::MatrixXd& z)
	{
		return static_cast<Derived*>(this)->train(z);
	}

protected:
	PCABase() {}
};

#endif // PCA_BASE_H