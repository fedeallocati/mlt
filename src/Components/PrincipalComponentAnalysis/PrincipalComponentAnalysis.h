#ifndef PRINCIPAL_COMPONENT_ANALYSIS_H
#define PRINCIPAL_COMPONENT_ANALYSIS_H

#include <EigenOptimization/Optimization>
#include <Eigen/SVD>

class PrincipalComponentAnalysis
{
public:
	PrincipalComponentAnalysis();
	PrincipalComponentAnalysis(const Eigen::MatrixXd& trainingSet, size_t k = 0);

	void train(const Eigen::MatrixXd& trainingSet, size_t k = 0);

	const size_t getNumberOfFeatures() const;

	const Eigen::MatrixXd& getMatrixUReduce() const;
	void setMatrixUReduce(const Eigen::MatrixXd&);

	Eigen::MatrixXd projectData(const Eigen::MatrixXd& x);
	Eigen::MatrixXd recoverData(const Eigen::MatrixXd& z);

private:
	bool trained;
	Eigen::MatrixXd matrixUReduce;
};

#endif // PRINCIPAL_COMPONENT_ANALYSIS_Hs