#define EIGEN_USE_MKL_ALL

#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>

#include <Eigen/Core>

//#include "../../mlt/models/transformers/autoencoder.hpp"
//#include "../../mlt/models/transformers/sparse_autoencoder.hpp"
//#include "../../mlt/models/transformers/tied_autoencoder.hpp"
//#include "../../mlt/models/transformers/sparse_tied_autoencoder.hpp"
#include "../../mlt/models/classifiers/optimizable_linear_classifier.hpp"
//#include "../../mlt/models/pipeline.hpp"
#include "../../mlt/utils/optimizers/stochastic_gradient_descent.hpp"
#include "../../mlt/utils/loss_functions.hpp"
#include "../misc.hpp"

std::vector<std::vector<double>> parseCsv(std::string file, bool skipFirstLine, char delim = ',') {
	std::vector<std::vector<double>> result;

	std::ifstream str(file.c_str());
	std::string line;

	if (skipFirstLine) {
		std::getline(str, line);
	}

	while (std::getline(str, line)) {
		std::vector<double> currentLine;
		std::stringstream lineStream(line);
		std::string cell;

		while (std::getline(lineStream, cell, delim)) {
			currentLine.push_back(std::stod(cell));
		}

		result.push_back(currentLine);
	}

	return result;
}

std::tuple<Eigen::MatrixXd, Eigen::VectorXi> load_training_data(std::string file) {
	auto data = parseCsv(file, true);
	Eigen::MatrixXd features(data[0].size(), data.size());
	Eigen::VectorXi classes(data.size());

	for(auto i = 0; i < data.size(); i++) {
		classes(i) = data[i][0];
		for (auto j = 1; j < data[i].size(); j++) {
			features(j - 1, i) = data[i][j];
		}
	}

	return{ features, classes };
}

Eigen::MatrixXd load_test_data(std::string file) {
	auto data = parseCsv(file, true);
	Eigen::MatrixXd features(data[0].size(), data.size());

	for (auto i = 0; i < data.size(); i++) {
		for (auto j = 0; j < data[i].size(); j++) {
			features(j, i) = data[i][j];
		}
	}

	return features;
}

void output_result(const Eigen::VectorXi& result, std::string filename)
{
	std::ofstream outputFile(filename, std::ofstream::app);

	outputFile << "ImageId,Label" << std::endl;

	for (size_t i = 0; i < result.rows(); i++) {
		outputFile << i + 1 << "," << result(i) << std::endl;
	}

	outputFile.close();

}

template <class Model, class TargetType>
double split_crossvalidation(Model& model, const Eigen::MatrixXd& features, const TargetType& target, double training_percentage) {
	assert(features.cols() == target.cols() || (target.cols() == 1 && features.cols() == target.rows()));
	assert(training_percentage > 0 && training_percentage < 1);
	auto training = static_cast<size_t>(std::round(features.cols() * training_percentage));
	auto validation = features.cols() - training;
	assert(training > 0 && validation > 0);

	auto training_features = features.leftCols(training);
	auto training_target = target.cols() == 1 ? target.block(0, 0, training, 1) : target.block(0, 0, target.rows(), training);
	auto validation_features = features.rightCols(validation);
	auto validation_target = target.cols() == 1 ? target.block(training, 0, validation, 1) : target.block(0, training, target.rows(), validation);

	model.fit(training_features, training_target);

	return model.score(validation_features, validation_target);
}

int main() {
	print_info();
	std::cout << std::endl;

	Eigen::MatrixXd features;
	Eigen::VectorXi classes;

	std::tie(features, classes) = load_training_data("train.csv");
	
	using loss1_t = mlt::utils::loss_functions::SoftmaxLoss;
	using opt_t = mlt::utils::optimizers::StochasticGradientDescent<>;
	
	for (auto batch_size : {0, 64, 128, 256, 512, 1024}) {
		for (auto learning_rate : { 0.1, 0.01, 0.001, 0.001}) {
			for (auto decay : { 0.99, 0.95 }) {
				for (auto regularization : { 0.005, 0.05, 0.5, 1.0, 5.0 }) {

					loss1_t loss1;
					opt_t opt(batch_size, 200, learning_rate, decay);

					mlt::models::classifiers::OptimizableLinearClassifier<loss1_t, opt_t> model1(loss1, opt, regularization, true);

					double score = split_crossvalidation(model1, features, classes, 0.8);

					std::cout << "Score for: " << batch_size << " " << learning_rate << " " << decay << " " << regularization << ": " << score << std::endl;
				}
			}
		}
	}	

	std::cin.get();

	return 0;
}