#define EIGEN_USE_MKL_ALL

#include <iostream>

#include <Eigen/Core>

#include "../misc.hpp"

#include "models/transformers/principal_components_analysis.hpp"
#include "models/transformers/zero_components_analysis.hpp"

void pca_examples() {
	Eigen::MatrixXd X(2, 6);

	X.row(0) << -1, -2, -3, 1, 2, 3;
	X.row(1) << -1, -1, -2, 1, 1, 2;

	mlt::models::transformers::PrincipalComponentsAnalysis pca;
	pca.fit(X);

	std::cout << "PCA: " << std::endl;
	std::cout << pca.explained_variance_ratio() << std::endl << std::endl;
	std::cout << "X2" << std::endl << X << std::endl;
	std::cout << "PCA(X2)" << std::endl << pca.transform(X) << std::endl;
	std::cout << "PCA-1(PCA(X2))" << std::endl << pca.inverse_transform(pca.transform(X)) << std::endl;
	std::cout << "PCA(X2[:,1])" << std::endl << pca.transform(X.leftCols(1)) << std::endl;
	std::cout << "PCA-1(PCA(X2[:,1]))" << std::endl << pca.inverse_transform(pca.transform(X.leftCols(1))) << std::endl << std::endl;

	mlt::models::transformers::ZeroComponentsAnalysis zca;
	zca.fit(X);

	std::cout << "ZCA: " << std::endl;
	std::cout << zca.explained_variance_ratio() << std::endl << std::endl;
	std::cout << "X2" << std::endl << X << std::endl;
	std::cout << "ZCA(X2)" << std::endl << zca.transform(X) << std::endl;
	std::cout << "ZCA-1(ZCA(X2))" << std::endl << zca.inverse_transform(zca.transform(X)) << std::endl;
	std::cout << "ZCA(X2[:,1])" << std::endl << zca.transform(X.leftCols(1)) << std::endl;
	std::cout << "ZCA-1(ZCA(X2[:,1]))" << std::endl << zca.inverse_transform(zca.transform(X.leftCols(1))) << std::endl << std::endl;
}