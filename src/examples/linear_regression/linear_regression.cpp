#define EIGEN_VECTORIZE_SSE4_2
#define EIGEN_USE_MKL_ALL

#include <vector>
#include <chrono>
#include <iostream>
#include <iomanip>

#include <Eigen/Core>
#include <omp.h>

#include "../../examples/misc/misc.hpp"
#include "../../mlt/models/regressors/linear_regressor.hpp"

using namespace std;
using namespace Eigen;

inline VectorXd correletedData(double x) {
	return Vector4d(1, x, 2 * x, x*x);
}

inline double correletedTarget(double x) {
	return 5 * x + 3;
}

struct Params {
	struct LinearRegression {
		static const int size = 2;
	};
};

int main() {
	cout << "#Threads: " << nbThreads() << endl;
	cout << "SIMD Instruction Sets In Use: " << SimdInstructionSetsInUse() << endl;

#ifdef EIGEN_USE_MKL
	cout << "MKL Enabled. Version: " << INTEL_MKL_VERSION << endl;
#endif	

	auto input_v = parse_csv<double>("house_data.csv", ',');
	MatrixXd input(input_v.size(), Params::LinearRegression::size + 1);
	VectorXd target(input_v.size(), 1);

	auto i = 0;
	for (const vector<double>& row : input_v) {
		input(i, 0) = 1;
		for (auto j = 0; j < Params::LinearRegression::size; j++) {
			input(i, j + 1) = row[j];
		}

		target(i) = row[Params::LinearRegression::size];
		i++;
	}

	/*VectorXd points = VectorXd::Random(10000000, 1);
	MatrixXd input(points.rows(), Params::LinearRegression::size + 1);
	VectorXd target(points.rows(), 1);

	for (auto i = 0; i < points.rows(); i++) {
	input.row(i) = correletedData(points(i));
	target(i) = correletedTarget(points(i));
	}*/

	cout << std::setprecision(10) << endl;

	cout << "First 10 examples from the dataset: " << endl;
	for (auto i = 0; i < 10; i++) {
		cout << "x = [" << input.row(i) << "], y = " << target(i) << endl;
	}

	mlt::models::regressors::LinearRegressor<Params> lr;
	cout << "Train Time: " << benchmark<std::chrono::milliseconds>([&]() { lr.self_train(input, target); }).count() << "ms\n Theta: \n" << lr.params() << endl << endl;

	Vector3d test(1.0, 1650.0, 3.0);
	cout << lr.regress(test) << endl;


	cin.get();
	return 0;
}