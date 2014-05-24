#include "PrincipalComponentAnalysis.h"

PrincipalComponentAnalysis::PrincipalComponentAnalysis() : trained(false)
{
}

PrincipalComponentAnalysis::PrincipalComponentAnalysis(const Eigen::MatrixXd& trainingSet, size_t k)
{
	this->train(trainingSet, k);
}

void PrincipalComponentAnalysis::train(const Eigen::MatrixXd& trainingSet, size_t k)
{
	eigen_assert(k <= trainingSet.cols());
	Eigen::JacobiSVD<Eigen::MatrixXd> svd = ((trainingSet.transpose() * trainingSet) / trainingSet.rows()).jacobiSvd(Eigen::ComputeThinU);

	if (k == 0)
	{
		double sum = svd.singularValues().sum();
		double acum = 0;

		for(size_t i = 0; i < svd.singularValues().rows(); i++)
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

const size_t PrincipalComponentAnalysis::getNumberOfFeatures() const
{
	eigen_assert(this->trained);
	return this->matrixUReduce.cols();
}

const Eigen::MatrixXd& PrincipalComponentAnalysis::getMatrixUReduce() const
{
	eigen_assert(this->trained);
	return this->matrixUReduce;
}

void PrincipalComponentAnalysis::setMatrixUReduce(const Eigen::MatrixXd& matrixUReduce)
{
	this->matrixUReduce = matrixUReduce;
	this->trained = true;
}

Eigen::MatrixXd PrincipalComponentAnalysis::projectData(const Eigen::MatrixXd& x)
{
	eigen_assert(this->trained);
	return x * this->matrixUReduce;
}

Eigen::MatrixXd PrincipalComponentAnalysis::recoverData(const Eigen::MatrixXd& z)
{
	eigen_assert(this->trained);
	return z * this->matrixUReduce.transpose();
}