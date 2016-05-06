#ifndef MLT_UTILS_EIGEN_HPP
#define MLT_UTILS_EIGEN_HPP

#include <random>

#include <Eigen/Core>

#include "types.hpp"

namespace mlt {
namespace utils {
namespace eigen {
	using namespace Eigen;

	inline Map<const MatrixXd> ravel(const Ref<const MatrixXd>& x) {
		return Map<const MatrixXd>(x.data(), x.size(), 1);
	}

	inline Map<const MatrixXd> unravel(const Ref<const MatrixXd>& x, size_t rows, size_t cols) {
		return Map<const MatrixXd>(x.data(), rows, cols);
	}

	template <typename ScalarA, int RowsA, int ColsA, typename ScalarB, int RowsB, int ColsB, typename URng = std::default_random_engine >
	inline std::tuple<Matrix<ScalarA, Dynamic, Dynamic>, Matrix<ScalarB, Dynamic, Dynamic>>
	tied_random_cols_subset(const Ref<const Matrix<ScalarA, RowsA, ColsA>>& a, const Ref<const Matrix<ScalarB, RowsB, ColsB>>& b, size_t subset_size, URng&& rng = URng()) {
		assert(subset_size > 0);
		std::uniform_int_distribution<size_t> distribution(0, a.cols() - 1);

		auto a_batch = Matrix<ScalarA, Dynamic, Dynamic>(a.rows(), subset_size);
		auto b_batch = Matrix<ScalarB, Dynamic, Dynamic>(b.rows(), subset_size);

		for (auto i = 0; i < subset_size; i++) {
			auto cidx = distribution(rng);
			a_batch.col(i) = a.col(cidx);
			b_batch.col(i) = b.col(cidx);
		}

		return{ a_batch, b_batch };
	}

	template <typename ScalarA, int RowsA, int ColsA, typename ScalarB, int RowsB, int ColsB, typename URng = std::default_random_engine >
	inline std::tuple<Matrix<ScalarA, Dynamic, Dynamic>, Matrix<ScalarB, Dynamic, Dynamic>>
	tied_random_cols_subset(const Matrix<ScalarA, RowsA, ColsA>& a, const Matrix<ScalarB, RowsB, ColsB>& b, size_t subset_size, URng&& rng = URng()) {
		assert(subset_size > 0);
		std::uniform_int_distribution<size_t> distribution(0, a.cols() - 1);

		auto a_batch = Matrix<ScalarA, Dynamic, Dynamic>(a.rows(), subset_size);
		auto b_batch = Matrix<ScalarB, Dynamic, Dynamic>(b.rows(), subset_size);

		for (auto i = 0; i < subset_size; i++) {
			auto cidx = distribution(rng);
			a_batch.col(i) = a.col(cidx);
			b_batch.col(i) = b.col(cidx);
		}

		return{ a_batch, b_batch };
	}
	
}
}
}
#endif