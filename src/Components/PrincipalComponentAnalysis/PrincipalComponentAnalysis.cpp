#include "PrincipalComponentAnalysis.h"

PrincipalComponentAnalysis::PrincipalComponentAnalysis(const Eigen::MatrixXd& trainingSet, size_t k)
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
}

Eigen::MatrixXd PrincipalComponentAnalysis::projectData(const Eigen::MatrixXd& x)
{
	return x * this->matrixUReduce;
}

Eigen::MatrixXd PrincipalComponentAnalysis::recoverData(const Eigen::MatrixXd& z)
{
	return z * this->matrixUReduce.transpose();
}