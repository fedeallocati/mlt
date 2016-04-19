#define EIGEN_USE_MKL_ALL

#include <chrono>

#include <Eigen/Core>

//#include "datasets.hpp"
#include "../mlt/models/regressors/least_squares_linear_regression.hpp"
#include "../mlt/models/regressors/ridge_regression.hpp"
#include "../mlt/models/regressors/optimizable_linear_regressor.hpp"
#include "../mlt/utils/loss_functions.hpp"
#include "../mlt/models/transformers/principal_components_analysis.hpp"
#include "../mlt/models/transformers/zero_components_analysis.hpp"
#include "../mlt/models/pipeline.hpp"

namespace regressors = mlt::models::regressors;
namespace linear_solvers = mlt::utils::linear_solvers;

//extern void lr_example(std::tuple<Eigen::MatrixXd, Eigen::MatrixXd> data, Eigen::VectorXd test);
//extern void pca_example(Eigen::MatrixXd data, Eigen::VectorXd test);

template <typename Model>
void benchmark(Model& model, Eigen::MatrixXd dataset, Eigen::MatrixXd target, int iters) {
	double min = std::numeric_limits<double>::max(), max = 0, total = 0;

	for (auto it = 0; it < iters; it++) {
		auto start = std::chrono::steady_clock::now();
		model.fit(dataset, target);
		auto end = std::chrono::steady_clock::now();
		double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();				
		min = std::min(min, elapsed);
		max = std::max(max, elapsed);
		total += elapsed;
	}

	std::cout << "Benchmark for " << typeid(Model).name() << " with input (" << dataset.rows() << ", " << dataset.cols() << ") for " << iters << " iterations" << std::endl;
	std::cout << "Min: " << min << "ms" << std::endl;
	std::cout << "Max: " << max << "ms" << std::endl;
	std::cout << "Avg: " << total / iters << "ms" << std::endl;
	std::cout << "Tot: " << total << "ms" << std::endl;
}

void benchmark_linear_solvers() {
	Eigen::MatrixXd XN = Eigen::MatrixXd::Random(100, 100000);
	Eigen::RowVectorXd YN = Eigen::RowVectorXd::Random(100000);

	regressors::LeastSquaresLinearRegression<linear_solvers::SVDSolver> linear_regressor_svd(false);
	benchmark(linear_regressor_svd, XN, YN, 100);
	regressors::LeastSquaresLinearRegression<linear_solvers::LDLTSolver> linear_regressor_ldlt(false);
	benchmark(linear_regressor_ldlt, XN, YN, 100);
	std::cout << std::endl;
	regressors::LeastSquaresLinearRegression<linear_solvers::CGSolver> linear_regressor_cg(false);
	benchmark(linear_regressor_cg, XN, YN, 100);
	std::cout << std::endl;

	std::cout << "Diff: " << (linear_regressor_svd.coefficients() - linear_regressor_ldlt.coefficients()).squaredNorm() << std::endl;
	std::cout << "Diff: " << (linear_regressor_svd.coefficients() - linear_regressor_cg.coefficients()).squaredNorm() << std::endl;
	std::cout << "Diff: " << (linear_regressor_ldlt.coefficients() - linear_regressor_cg.coefficients()).squaredNorm() << std::endl;	
}

int main() {
	/*lr_example(house_value_dataset(), Eigen::Vector3d(1, 1650, 3));
	lr_example(correlated_data_dataset(1000000), correlatedData(0));
	pca_example(std::get<0>(house_value_dataset()), Eigen::Vector3d(1, 1650, 3));
	pca_example(std::get<0>(correlated_data_dataset(1000000)), correlatedData(0));*/

	//benchmark_linear_solvers();

	std::cout << "Eigen Version: " << EIGEN_WORLD_VERSION << "." << EIGEN_MAJOR_VERSION << "." << EIGEN_MINOR_VERSION << std::endl;
	std::cout << "Eigen Threads: " << Eigen::nbThreads() << std::endl;
	std::cout << "Eigen SIMD Instruction Sets In Use: " << Eigen::SimdInstructionSetsInUse() << std::endl;

#ifdef EIGEN_USE_MKL
	std::cout << "MKL Version: " << __INTEL_MKL__ << "." << __INTEL_MKL_MINOR__ << "." << __INTEL_MKL_UPDATE__ << std::endl;
	std::cout << "MKL Threads: " << mkl_get_max_threads();
	if (MKL_Get_Dynamic())
		std::cout << " (may use less for large problems)";
	std::cout << std::endl;
#endif
	std::cout << std::endl;

	/*Eigen::MatrixXd X1(2, 3);
	Eigen::MatrixXd Y1(1, 3);

	X1.row(0) << 0, 1, 2;
	X1.row(1) << 0, 1, 2;
	Y1 << 0, 1, 2;

	regressors::LeastSquaresLinearRegression<> linear_regressor(false);
	linear_regressor.fit(X1, Y1);

	std::cout << "LinearRegression: " << std::endl;
	std::cout << linear_regressor.coefficients() << std::endl;
	if (linear_regressor.fit_intercept()) {
		std::cout << linear_regressor.intercepts() << std::endl;
	}
	std::cout << linear_regressor.predict(X1.col(0)) << std::endl;

	Eigen::MatrixXd X2(2, 3);
	Eigen::MatrixXd Y2(1, 3);

	X2.row(0) << 0, 0, 1;
	X2.row(1) << 0, 0, 1;
	Y2 << 0, .1, 1;

	regressors::RidgeRegression<> ridge_regressor(0.5, true);
	ridge_regressor.fit(X2, Y2);

	std::cout << "RidgeRegression: " << std::endl;
	std::cout << ridge_regressor.coefficients() << std::endl;
	std::cout << ridge_regressor.intercepts() << std::endl;
	std::cout << ridge_regressor.predict(X2.col(0)) << std::endl;

	Eigen::MatrixXd XN = Eigen::MatrixXd::Random(100, 100000);
	Eigen::MatrixXd YN = Eigen::MatrixXd::Random(3, 100000);

	auto pred = ridge_regressor.fit(XN, YN).predict(XN.leftCols<2>());
	std::cout << "Expected size: 3x2. Obtained size: " << pred.rows() << "x" << pred.cols() << std::endl;

	Eigen::MatrixXd X3(2, 6);

	X3.row(0) << -1, -2, -3, 1, 2, 3;
	X3.row(1) << -1, -1, -2, 1, 1, 2;

	mlt::models::transformers::PrincipalComponentsAnalysis pca;
	pca.fit(X3);

	std::cout << "PCA: " << std::endl;
	std::cout << pca.explained_variance_ratio() << std::endl << std::endl;
	std::cout << "X2" << std::endl << X3 << std::endl;
	std::cout << "PCA(X2)" << std::endl << pca.transform(X3) << std::endl;
	std::cout << "PCA-1(PCA(X2))" << std::endl << pca.inverse_transform(pca.transform(X3)) << std::endl;
	std::cout << "PCA(X2[:,1])" << std::endl << pca.transform(X3.leftCols(1)) << std::endl;
	std::cout << "PCA-1(PCA(X2[:,1]))" << std::endl << pca.inverse_transform(pca.transform(X3.leftCols(1))) << std::endl << std::endl;

	mlt::models::transformers::ZeroComponentsAnalysis zca;
	zca.fit(X3);

	std::cout << "ZCA: " << std::endl;
	std::cout << zca.explained_variance_ratio() << std::endl << std::endl;
	std::cout << "X2" << std::endl << X3 << std::endl;
	std::cout << "ZCA(X2)" << std::endl << zca.transform(X3) << std::endl;
	std::cout << "ZCA-1(ZCA(X2))" << std::endl << zca.inverse_transform(zca.transform(X3)) << std::endl;
	std::cout << "ZCA(X2[:,1])" << std::endl << zca.transform(X3.leftCols(1)) << std::endl;
	std::cout << "ZCA-1(ZCA(X2[:,1]))" << std::endl << zca.inverse_transform(zca.transform(X3.leftCols(1))) << std::endl << std::endl;*/

	/*auto pipeline = mlt::models::create_pipeline(mlt::models::transformers::ZeroComponentsAnalysis(),
		mlt::models::transformers::PrincipalComponentsAnalysis(0.99), ridge_regressor);*/

	//mlt::utils::optimizers::StochasticGradientDescent<mlt::utils::optimizers::RMSPropGradientDescentUpdate> grad_descent(10, 2000, 0.001, 1, mlt::utils::optimizers::RMSPropGradientDescentUpdate());
	/*mlt::models::regressors::SquaredLoss<> sgd(0.0,  false);
	regressors::RidgeRegression<> ridge_regressor2(0.0, false);

	ridge_regressor2.fit(input, output);
	sgd.fit(input, output, false);
	
	std::cout << "RidgeRegression: " << std::endl;
	std::cout << ridge_regressor2.predict(input) << std::endl;
	std::cout << "SgdEaeapepe: " << std::endl;
	std::cout << sgd.predict(input) << std::endl;

	std::cout << "loss with closed form: " << sgd.loss(ridge_regressor2.coefficients(), input, output) << std::endl;
	std::cout << "loss with SGD: " << sgd.loss(sgd.coefficients(), input, output) << std::endl;*/

	mlt::utils::loss_functions::SoftmaxLoss loss1;
	mlt::utils::loss_functions::HingeLoss loss2(1.0);

	Eigen::MatrixXd input = Eigen::MatrixXd::Random(3, 10);
	Eigen::MatrixXd output = Eigen::MatrixXd::Random(2, 10);

	regressors::RidgeRegression<> ridge_regressor2(0.0, false);

	ridge_regressor2.fit(input, output);

	auto a1 = softmax_regression_cost_and_gradient(ridge_regressor2.coefficients().transpose(), input.transpose(), output.transpose());
	auto b1 = loss1.loss_and_gradient(ridge_regressor2.coefficients() * input, output);

	std::cout << std::get<0>(a1) << " vs " << std::get<0>(b1) << std::endl;
	std::cout << std::get<1>(a1).transpose() << std::endl << " vs " << std::endl << std::get<1>(b1) * input.transpose() << std::endl;

	auto a2 = linear_svm_cost_and_gradient(ridge_regressor2.coefficients().transpose(), input.transpose(), output.transpose());
	auto b2 = loss2.loss_and_gradient(ridge_regressor2.coefficients() * input, output);

	std::cout << std::get<0>(a2) << " vs " << std::get<0>(b2) << std::endl;
	std::cout << std::get<1>(a2).transpose() << std::endl << " vs " << std::endl << std::get<1>(b2) * input.transpose() << std::endl;

	std::cin.get();

	return 0;
}