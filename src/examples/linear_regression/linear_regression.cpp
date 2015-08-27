#define EIGEN_USE_MKL_ALL
#define MLT_VERBOSE_TRAINING

#include <iostream>
#include <iomanip>

#include <Eigen/Eigen>

#include "../../examples/misc/misc.hpp"
#include "../../mlt/models/regressors/least_squares_linear_regressor.hpp"
#include "../../mlt/trainers/gradient_based/gradient_descent.hpp"

using namespace std;
using namespace Eigen;

using namespace mlt::models::regressors;
using namespace mlt::trainers::gradient_based;

struct Params {
	struct GradientDescent {
		static constexpr int epochs = 400;
		static constexpr int batch_size = 0;
		static constexpr double learning_rate = 0.01;
		static constexpr double learning_rate_decay = 1;
		static constexpr gradient_descent_update_t update_method = gradient_descent_update_t::gradient_descent;
		static constexpr double update_param = 0;
	};
};

tuple<MatrixXd, MatrixXd> house_value_dataset() {
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
	
	return make_tuple(input, target);
}

inline VectorXd correletedData(double x) {
	return (VectorXd(8) << 1, x, 2 * x, x*x, 5, 6, 3 * x, 0.5 * x).finished();
}

inline VectorXd correletedTarget(double x) {
	return Vector2d(5 * x + 3, -2 * x - 10);
}

tuple<MatrixXd, MatrixXd> correlated_data_dataset(int n) {	
	VectorXd points = VectorXd::Random(n, 1) * 100;
	MatrixXd input(points.rows(), 3);
	MatrixXd target(points.rows(), 2);

	for (auto i = 0; i < points.rows(); i++) {
		input.row(i) = correletedData(points(i)).topRows(input.cols());
		target.row(i) = correletedTarget(points(i));
	}

	return make_tuple(input, target);
}

int main() {
	print_info();

	MatrixXd input, target;

	std::tie(input, target) = house_value_dataset();

	// Normalize input features
	input.rightCols(input.cols() - 1).rowwise() -= input.rightCols(input.cols() - 1).colwise().mean();
	MatrixXd cov = (input.rightCols(input.cols() - 1).adjoint() * input.rightCols(input.cols() - 1)) / double(input.rows() - 1);
	input.array().rightCols(input.cols() - 1).rowwise() /= cov.diagonal().transpose().array().sqrt();

	cout << std::setprecision(6) << endl;

	LeastSquaresLinearRegressor LSLR_t;

	LeastSquaresLinearRegressor lr1(input.cols() - 1, target.cols());
	GradientDescentTrainer<Params, LeastSquaresLinearRegressor> gdt(lr1);
	
	auto time = benchmark([&]() { gdt.train(input, target); }).count();
	
	cout << "Train Time: " << time << "ms\nTheta: \n" << lr1.params() << endl << endl;
	cout << "Cost: " << lr1.cost(input, target) << endl;
	cout << "Cost Gradient: " << endl;
	cout << lr1.cost_gradient(input, target) << endl << endl;

	LeastSquaresLinearRegressor lr2;
	auto time2 = benchmark([&]() { lr2.self_train(input, target); }).count();
	cout << "Train Time: " << time2 << "ms" << endl;
	cout << "Theta: \n" << lr2.params() << endl << endl;

	cout << "Cost: " << lr2.cost(input, target) << endl;	
	cout << "Cost Gradient: " << endl;
	cout << lr2.cost_gradient(input, target) << endl;
		
	cin.get();

	return 0;
}