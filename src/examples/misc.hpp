#ifndef EXAMPLES_MISC_HPP
#define EXAMPLES_MISC_HPP

#include <iostream>
#include <type_traits>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <chrono>

#include <Eigen/Core>

inline void print_info() {
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
}

void pca_examples();

void lr_examples();

template <typename T>
typename std::enable_if<std::is_floating_point<T>::value, std::vector<std::vector<T> > >::type parse_csv(std::string file, char separator = ',') {
    std::vector<std::vector<T>> result;
    std::ifstream fin(file.c_str());
    
    int row_length = -1;

    for (std::string line; getline(fin, line);) {
        auto comment = line.find_first_of('#');
        if (comment != line.npos) {
            line = line.substr(0, comment);
        }

        std::vector<T> row;
        if (row_length > 0) {
            row.reserve(row_length);
        }

        std::istringstream line_stream(line);
        std::string cell;

        while (getline(line_stream, cell, separator)) {
            if (cell.find('.') == cell.npos){
                row.push_back(atoi(cell.c_str()));
            } else {
                row.push_back(atof(cell.c_str()));
            }
        }

        if (row.size() > 0) {
            result.push_back(row);
            row_length = row.size();
        }
    }

    return result;
}

template <typename T>
typename std::enable_if<std::is_integral<T>::value, std::vector<std::vector<T> > >::type parse_csv(std::string file, char separator = ',') {
    std::vector<std::vector<T>> result;
    std::ifstream fin(file.c_str());

    int row_length = -1;

    for (std::string line; getline(fin, line);) {
        auto comment = line.find_first_of('#');
        if (comment != line.npos) {
            line = line.substr(0, comment);
        }

        std::vector<T> row;
        if (row_length > 0) {
            row.reserve(row_length);
        }

        std::istringstream line_stream(line);
        std::string cell;

        while (getline(line_stream, cell, separator)) {
            row.push_back(atoi(cell.c_str()));
        }

        if (row.size() > 0) {
            result.push_back(row);
            row_length = row.size();
        }
    }

    return result;
}

template <class T = std::chrono::milliseconds, typename F>
T benchmark_single(const F& f) {
    auto t1 = std::chrono::steady_clock::now();

    f();

    auto t2 = std::chrono::steady_clock::now();

    return std::chrono::duration_cast<T>(t2 - t1);
}

template <class Model>
void benchmark(Model& model, Eigen::MatrixXd dataset, Eigen::MatrixXd target, int iters) {
	double min = std::numeric_limits<double>::max(), max = 0, total = 0;

	for (auto it = 0; it < iters; it++) {
		double elapsed = benchmark_single<std::chrono::milliseconds>([&] { model.fit(dataset, target); }).count();
		min = std::min(min, elapsed);
		max = std::max(max, elapsed);
		total += elapsed;
	}

	std::cout << "Benchmark for " << typeid(Model).name();
	std::cout << " with input ([" << dataset.rows() << "*" << dataset.cols() << "], [" << target.rows() << "*" << target.cols() << "])";
	std::cout << " for " << iters << " iterations" << std::endl;
	std::cout << "Min: " << min << "ms" << std::endl;
	std::cout << "Max: " << max << "ms" << std::endl;
	std::cout << "Avg: " << total / iters << "ms" << std::endl;
	std::cout << "Tot: " << total << "ms" << std::endl;
}

template <class Model>
void benchmark(Model& model, Eigen::MatrixXd dataset, int iters) {
	std::cout << "Benchmark for " << typeid(Model).name();
	std::cout << " with input ([" << dataset.rows() << "*" << dataset.cols() << "])";
	std::cout << " for " << iters << " iterations" << std::endl;

	double min = std::numeric_limits<double>::max(), max = 0, total = 0;

	for (auto it = 0; it < iters; it++) {
		double elapsed = benchmark_single<std::chrono::milliseconds>([&] { model.fit(dataset, target); }).count();
		min = std::min(min, elapsed);
		max = std::max(max, elapsed);
		total += elapsed;
	}

	std::cout << "Min: " << min << "ms" << std::endl;
	std::cout << "Max: " << max << "ms" << std::endl;
	std::cout << "Avg: " << total / iters << "ms" << std::endl;
	std::cout << "Tot: " << total << "ms" << std::endl;
}

template <class Model>
void eval_numerical_gradient(const Model& model, const Eigen::MatrixXd& params, const Eigen::MatrixXd& dataset, const Eigen::MatrixXd& target) {
	std::cout << "Numerical Gradient for " << typeid(Model).name();
	std::cout << " with input ([" << dataset.rows() << "*" << dataset.cols() << "], [" << target.rows() << "*" << target.cols() << "])" << std::endl;

	double fx = model.loss(params, dataset, target);
	Eigen::MatrixXd adfx = model.gradient(params, dataset, target);
	Eigen::MatrixXd ndfx = Eigen::MatrixXd::Zero(adfx.rows(), adfx.cols());
	auto h = 0.000001;

	Eigen::MatrixXd params_copy = params;

	for (auto i = 0; i < params.rows(); i++) {
		for (auto j = 0; j < params.cols(); j++) {
			params_copy(i, j) += h;
			double fxh = model.loss(params_copy, dataset, target);
			params_copy(i, j) -= h;
			ndfx(i, j) = (fxh - fx) / h;
		}
	}

	Eigen::MatrixXd side(params.size(), 3);
	side.col(0) = mlt::utils::eigen::ravel(adfx);
	side.col(1) = mlt::utils::eigen::ravel(ndfx);
	side.col(2) = mlt::utils::eigen::ravel((ndfx - adfx).cwiseAbs().cwiseQuotient(((ndfx.cwiseAbs() + adfx.cwiseAbs()).array() + 1e-5).matrix()));
	std::cout << side << std::endl;
}

#endif