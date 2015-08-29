#define EIGEN_USE_MKL_ALL
//#define MLT_VERBOSE_TRAINING

#include <iostream>
#include <iomanip>

#include <Eigen/Eigen>

#include "misc.hpp"
#include "../mlt/models/transformations/principal_components_analysis.hpp"

using namespace std;
using namespace Eigen;

using namespace mlt::models::transformations;

struct Params {
	struct PrincipalComponentsAnalysis {
		static constexpr bool normalize_mean = true;
		static constexpr bool normalize_variance = true;		
		static constexpr int new_dimension = 2;
		static constexpr double variance_to_retain = 0;		
	};
};

void pca_example(MatrixXd input, VectorXd test) {
	PrincipalComponentsAnalysis<Params> pca;
		
	cout << "Training Principal Components Analysis.." << endl;
	auto time = benchmark([&]() { pca.self_train(input); }).count();

	cout << endl;
	cout << "Train Time: \t" << time << "ms" << endl << endl;

	cout << "Matrix U Found: " << endl << pca.matrix_u() << endl << endl;

	VectorXd test_norm = test;
	
	cout << "Transformation for test: " << endl << pca.transform_single(test_norm) << endl << endl;
	cin.get();
}