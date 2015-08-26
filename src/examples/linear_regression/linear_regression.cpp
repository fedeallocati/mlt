//#define EIGEN_VECTORIZE_SSE4_2
#define EIGEN_USE_MKL_ALL

#include <iostream>
#include <iomanip>

#include <Eigen/Eigen>

#include "../../examples/misc/misc.hpp"
#include "../../mlt/models/regressors/least_squares_linear_regressor.hpp"
#include "../../mlt/trainers/gradient_based/gradient_descent.hpp"

using namespace std;
using namespace Eigen;

inline VectorXd correletedData(double x) {
	return (VectorXd(8) << 1, x, 2 * x, x*x, 5, 6, 3 * x, 0.5 * x).finished();
}

inline double correletedTarget(double x) {
	return 5 * x + 3;
}

struct Params {
	struct GradientDescent {
		static constexpr int epochs = 400;
		static constexpr int batch_size = 0;
		static constexpr double learning_rate = 0.01;
		static constexpr double learning_rate_decay = 1;
		static constexpr double momentum = 0.9;
		static constexpr bool nesterov_momentum = true;		
	};
};

int main() {
	print_info();

	/*VectorXd points = VectorXd::Random(10000000, 1);
	MatrixXd input(points.rows(), 8);
	VectorXd target(points.rows(), 1);

	for (auto i = 0; i < points.rows(); i++) {
		input.row(i) = correletedData(points(i));
		target(i) = correletedTarget(points(i));
	}*/

	auto input_v = parse_csv<double>("house_data.csv", ',');
	MatrixXd input(input_v.size(), 3);
	VectorXd target(input_v.size(), 1);

	auto i = 0;
	for (const vector<double>& row : input_v) {
		input(i, 0) = 1;
		for (auto j = 0; j < 2; j++) {
			input(i, j + 1) = row[j];
		}
		target(i) = row[2];
		i++;
	}

	input.rightCols(2).rowwise() -= input.rightCols(2).colwise().mean();
	MatrixXd cov = (input.rightCols(2).adjoint() * input.rightCols(2)) / double(input_v.size() - 1);	
	input.array().rightCols(2).rowwise() /= cov.diagonal().transpose().array().sqrt();

	cout << std::setprecision(6) << endl;

	typedef mlt::models::regressors::LeastSquaresLinearRegressor LSLR_t;

	LSLR_t lr1(2);
	mlt::trainers::gradient_based::GradientDescentTrainer<Params, LSLR_t> gdt(lr1);
	
	long long time = benchmark([&]() { gdt.train(input, target); }).count();
	
	cout << "Train Time: " << time << "ms\nTheta: \n" << lr1.params() << endl << endl;
	cout << "Cost: " << lr1.cost(input, target) << endl;
	cout << "Cost Gradient: " << endl;
	cout << lr1.cost_gradient(input, target) << endl << endl;

	mlt::models::regressors::LeastSquaresLinearRegressor lr2;
	time = benchmark([&]() { lr2.self_train(input, target); }).count();
	cout << "Train Time: " << time << "ms" << endl;
	cout << "Theta: \n" << lr2.params() << endl << endl;

	cout << "Cost: " << lr2.cost(input, target) << endl;	
	cout << "Cost Gradient: " << endl;
	cout << lr2.cost_gradient(input, target) << endl;
		
	cin.get();
	return 0;
}