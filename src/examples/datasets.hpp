#ifndef DATASETS_HPP
#define DATASETS_HPP

#include <Eigen/Core>
#include <vector>

#include "misc.hpp"

std::tuple<Eigen::MatrixXd, Eigen::MatrixXd> house_value_dataset() {
	auto input_v = parse_csv<double>("house_data.csv", ',');
	Eigen::MatrixXd input(input_v.size(), 3);
	Eigen::VectorXd target(input_v.size(), 1);

	auto i = 0;
	for (const auto& row : input_v) {
		input(i, 0) = 1;
		for (auto j = 0; j < 2; j++) {
			input(i, j + 1) = row[j];
		}
		target(i) = row[2];
		i++;
	}

	return std::make_tuple(input, target);
}

inline Eigen::VectorXd correlatedData(double x) {
	return (Eigen::VectorXd(4) << 1, x, 2 * x, 0.5 * x*x).finished();
}

inline Eigen::VectorXd correlatedTarget(double x) {
	return (Eigen::VectorXd(2) << 5 * x + 3, x).finished();
}

std::tuple<Eigen::MatrixXd, Eigen::MatrixXd> correlated_data_dataset(int n) {
	Eigen::VectorXd points = Eigen::VectorXd::Random(n, 1) * 100;
	Eigen::MatrixXd input(points.rows(), correlatedData(1).rows());
	Eigen::MatrixXd target(points.rows(), correlatedTarget(1).rows());

	for (auto i = 0; i < points.rows(); i++) {
		input.row(i) = correlatedData(points(i)).topRows(input.cols());
		target.row(i) = correlatedTarget(points(i));
	}

	return std::make_tuple(input, target);
}
#endif