#ifndef MLT_UTILS_EIGEN_HPP
#define MLT_UTILS_EIGEN_HPP

#include <random>

#include <Eigen/Core>

#include "../defs.hpp"

namespace mlt {
namespace utils {
namespace eigen {
	inline Map<const MatrixXd> ravel(MatrixXdRef x) {
		return Map<const MatrixXd>(x.data(), x.size(), 1);
	}

	inline Map<const MatrixXd> unravel(MatrixXdRef x, size_t rows, size_t cols) {
		return Map<const MatrixXd>(x.data(), rows, cols);
	}

	template <class MatrixA, class MatrixB, typename Rng = default_random_engine >
	auto tied_random_cols_subset(MatrixA&& a, MatrixB&& b, size_t subset_size, Rng&& rng = Rng()) {
		assert(subset_size > 0);
		uniform_int_distribution<size_t> distribution(0, a.cols() - 1);

		auto a_batch = a.leftCols(subset_size).eval();
		auto b_batch = b.leftCols(subset_size).eval();

		for (auto i = 0; i < subset_size; i++) {
			auto cidx = distribution(rng);
			a_batch.col(i) = a.col(cidx);
			b_batch.col(i) = b.col(cidx);
		}

		return make_tuple(a_batch, b_batch);
	}

	auto classes_vector_to_classes_matrix(VectorXiRef classes) {
		auto classes_matrix = MatrixXi{ MatrixXi::Zero(classes.maxCoeff() + 1, classes.size()) };

		for (unsigned int i = 0; i < classes.size(); i++) {
			classes_matrix(classes(i), i) = 1;
		}

		return classes_matrix;
	}

	auto classes_matrix_to_classes_vector(MatrixXiRef classes) {
		assert((classes.colwise().sum().array() == 1).all());

		auto classes_vector = VectorXi(classes.cols());

		for (size_t col = 0; col < classes.cols(); col++) {
			for (size_t row = 0; row < classes.rows(); row++) {
				if (classes(row, col) == 1) {
					classes_vector(col) = (row);
					break;
				}
			}
		}

		return classes_vector;
	}

	inline auto max_row(VectorXdRef x) {
		int max_row;
		x.maxCoeff(&max_row);
		return max_row;
	}
}
}
}
#endif