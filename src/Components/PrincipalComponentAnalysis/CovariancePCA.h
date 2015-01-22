#ifndef COVARIANCE_PCA_H
#define COVARIANCE_PCA_H

#include <Eigen/SVD>
#include "PCABase.h"

class CovariancePCA : public PCABase<CovariancePCA>
{
public:
	CovariancePCA() : trained(false) {}

	CovariancePCA(const Eigen::MatrixXd& matrixUReduce) : matrixUReduce(matrixUReduce), trained(true) {}

	void train(const Eigen::MatrixXd& trainingSet, size_t k = 0)
	{
		eigen_assert(k <= trainingSet.cols());
		Eigen::JacobiSVD<Eigen::MatrixXd> svd = ((trainingSet.transpose() * trainingSet) / trainingSet.rows()).jacobiSvd(Eigen::ComputeThinU);

		if (k == 0)
		{
			double sum = svd.singularValues().sum();
			double acum = 0;

			for (size_t i = 0; i < svd.singularValues().rows(); i++)
			{
				acum += svd.singularValues()(i);

				if ((acum / sum) >= 0.99)
				{
					k = i;
					break;
				}
			}
		}

		this->matrixUReduce = svd.matrixU().leftCols(k);
		this->trained = true;
	}

	Eigen::MatrixXd transform(const Eigen::MatrixXd& x)
	{
		eigen_assert(this->trained);
		return x * this->matrixUReduce;
	}

	Eigen::MatrixXd recover(const Eigen::MatrixXd& z)
	{
		eigen_assert(this->trained);
		return z * this->matrixUReduce.transpose();
	}

	const size_t getNumberOfFeatures() const
	{
		eigen_assert(this->trained);
		return this->matrixUReduce.cols();
	}

	const Eigen::MatrixXd& getMatrixUReduce() const
	{
		eigen_assert(this->trained);
		return this->matrixUReduce;
	}

	void setMatrixUReduce(const Eigen::MatrixXd&)
	{
		this->matrixUReduce = matrixUReduce;
		this->trained = true;
	}	

private:
	bool trained;
	Eigen::MatrixXd matrixUReduce;
};

#endif // COVARIANCE_PCA_H