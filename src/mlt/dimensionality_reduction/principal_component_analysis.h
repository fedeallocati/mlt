#ifndef COVARIANCE_PCA_H
#define COVARIANCE_PCA_H

#include <Eigen/Core>
#include <Eigen/SVD>

#include "../base/iself_unsupervised_trainable.h"

namespace MLT
{
namespace DimensionalityReduction
{
	using namespace Eigen;
	using namespace Base;

	class PrincipalComponentAnalysis : ISelfUnsupervisedTrainable
	{
	public:
		PrincipalComponentAnalysis() : trained(false) {}

		PrincipalComponentAnalysis(const MatrixXd& matrixUReduce) : matrixUReduce(matrixUReduce), trained(true) {}

		void train(const MatrixXd& trainingSet)
		{
			//eigen_assert(k <= trainingSet.cols());
			JacobiSVD<MatrixXd> svd = ((trainingSet.transpose() * trainingSet) / trainingSet.rows()).jacobiSvd(ComputeThinU);
			int k = 0;
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

		MatrixXd transform(const MatrixXd& x)
		{
			eigen_assert(this->trained);
			return x * this->matrixUReduce;
		}

		MatrixXd recover(const MatrixXd& z)
		{
			eigen_assert(this->trained);
			return z * this->matrixUReduce.transpose();
		}

		const size_t getNumberOfFeatures() const
		{
			eigen_assert(this->trained);
			return this->matrixUReduce.cols();
		}

		const MatrixXd& getMatrixUReduce() const
		{
			eigen_assert(this->trained);
			return this->matrixUReduce;
		}

		void setMatrixUReduce(const MatrixXd&)
		{
			this->matrixUReduce = matrixUReduce;
			this->trained = true;
		}	

	private:
		bool trained;
		MatrixXd matrixUReduce;
	};
}
}
#endif // COVARIANCE_PCA_H