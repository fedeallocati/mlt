#ifndef PRINCIPAL_COMPONENT_ANALYSIS_H
#define PRINCIPAL_COMPONENT_ANALYSIS_H

#include <EigenOptimization/Optimization>
#include <Eigen/SVD>

class PrincipalComponentAnalysis
{
public:
	PrincipalComponentAnalysis(const Eigen::MatrixXd& trainingSet, size_t k = 0);

	Eigen::MatrixXd projectData(const Eigen::MatrixXd& x);
	Eigen::MatrixXd recoverData(const Eigen::MatrixXd& z);

private:
	Eigen::MatrixXd matrixUReduce;
};

#endif // PRINCIPAL_COMPONENT_ANALYSIS_Hs