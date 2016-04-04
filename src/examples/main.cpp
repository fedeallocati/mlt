#define EIGEN_USE_MKL_ALL
//#define MLT_VERBOSE_TRAINING

#include <chrono>

#include <Eigen/Core>

//#include "datasets.hpp"
#include "../mlt/models/regressors/least_squares_linear_regression.hpp"
#include "../mlt/models/regressors/ridge_regression.hpp"
#include "../mlt/models/transformations/principal_components_analysis.hpp"
#include "../mlt/models/transformations/zero_components_analysis.hpp"

//extern void lr_example(std::tuple<Eigen::MatrixXd, Eigen::MatrixXd> data, Eigen::VectorXd test);
//extern void pca_example(Eigen::MatrixXd data, Eigen::VectorXd test);

template <typename Model>
void benchmark(Model& model, Eigen::MatrixXd dataset, int iters)
{
	double min = std::numeric_limits<double>::max(), max = 0, total = 0;

	for (auto it = 0; it < iters; it++) {
		auto start = std::chrono::steady_clock::now();
		model.fit(dataset);
		auto end = std::chrono::steady_clock::now();
		double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();				
		min = std::min(min, elapsed);
		max = std::max(max, elapsed);
		total += elapsed;
	}

	std::cout << "Benchmark for " << typeid(Model).raw_name() << " with input (" << dataset.rows() << ", " << dataset.cols() << ") for " << iters << " iterations" << std::endl;
	std::cout << "Min: " << min << "ms" << std::endl;
	std::cout << "Max: " << max << "ms" << std::endl;
	std::cout << "Avg: " << total / iters << "ms" << std::endl;
	std::cout << "Tot: " << total << "ms" << std::endl;
}

int main() {
    /*lr_example(house_value_dataset(), Eigen::Vector3d(1, 1650, 3));
    lr_example(correlated_data_dataset(1000000), correlatedData(0));
    pca_example(std::get<0>(house_value_dataset()), Eigen::Vector3d(1, 1650, 3));
    pca_example(std::get<0>(correlated_data_dataset(1000000)), correlatedData(0));*/

	Eigen::MatrixXd X1(2, 3);
	Eigen::MatrixXd Y1(1, 3);

	X1.row(0) << 0, 1, 2;
	X1.row(1) << 0, 1, 2;
	Y1 << 0, 1, 2;

	mlt::models::regressors::LeastSquaresLinearRegression linear_regressor(false);
	linear_regressor.fit(X1, Y1);

	std::cout << "LinearRegression: " << std::endl;
	std::cout << linear_regressor.coefficients() << std::endl;

	Eigen::MatrixXd X2(2, 3);
	Eigen::MatrixXd Y2(1, 3);

	X2.row(0) << 0, 0, 1;
	X2.row(1) << 0, 0, 1;
	Y2 << 0, .1, 1;

	mlt::models::regressors::RidgeRegression ridge_regressor(0.5, true);
	ridge_regressor.fit(X2, Y2);
	
	std::cout << "RidgeRegression: " << std::endl;
	std::cout << ridge_regressor.coefficients() << std::endl;
	std::cout << ridge_regressor.intercepts() << std::endl;
	
	Eigen::MatrixXd X3(2, 6);

	X3.row(0) << -1, -2, -3, 1, 2, 3;
	X3.row(1) << -1, -1, -2, 1, 1, 2;

	mlt::models::transformations::PrincipalComponentsAnalysis pca;
	pca.fit(X3);

	std::cout << "PCA: " << std::endl;
	std::cout << pca.explained_variance_ratio() << std::endl << std::endl;
	std::cout << "X2" << std::endl << X3 << std::endl;
	std::cout << "PCA(X2)" << std::endl << pca.transform(X3) << std::endl;
	std::cout << "PCA-1(PCA(X2))" << std::endl << pca.inverse_transform(pca.transform(X3)) << std::endl;
	std::cout << "PCA(X2[:,1])" << std::endl << pca.transform(X3.leftCols(1)) << std::endl;
	std::cout << "PCA-1(PCA(X2[:,1]))" << std::endl << pca.inverse_transform(pca.transform(X3.leftCols(1))) << std::endl << std::endl;

	mlt::models::transformations::ZeroComponentsAnalysis zca;
	zca.fit(X3);

	std::cout << "ZCA: " << std::endl;
	std::cout << zca.explained_variance_ratio() << std::endl << std::endl;
	std::cout << "X2" << std::endl << X3 << std::endl;
	std::cout << "ZCA(X2)" << std::endl << zca.transform(X3) << std::endl;
	std::cout << "ZCA-1(ZCA(X2))" << std::endl << zca.inverse_transform(zca.transform(X3)) << std::endl;
	std::cout << "ZCA(X2[:,1])" << std::endl << zca.transform(X3.leftCols(1)) << std::endl;
	std::cout << "ZCA-1(ZCA(X2[:,1]))" << std::endl << zca.inverse_transform(zca.transform(X3.leftCols(1))) << std::endl << std::endl;

	std::cin.get();

	return 0;
}