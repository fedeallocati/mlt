#define EIGEN_USE_MKL_ALL
//#define MLT_VERBOSE


#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <iomanip>

#include <Eigen/Core>

//#include "../../mlt/models/transformers/autoencoder.hpp"
#include "../../mlt/models/transformers/sparse_autoencoder.hpp"
//#include "../../mlt/models/transformers/tied_autoencoder.hpp"
//#include "../../mlt/models/transformers/sparse_tied_autoencoder.hpp"
#include "../../mlt/models/classifiers/optimizable_linear_classifier.hpp"
#include "../../mlt/utils/optimizers/stochastic_gradient_descent.hpp"
#include "../../mlt/utils/loss_functions.hpp"
#include "../../mlt/utils/activation_functions.hpp"
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
			features(j - 1, i) = data[i][j] / 255.0;
		}
	}

	return{ features, classes };
}

Eigen::MatrixXd load_test_data(std::string file) {
	auto data = parseCsv(file, true);
	Eigen::MatrixXd features(data[0].size(), data.size());

	for (auto i = 0; i < data.size(); i++) {
		for (auto j = 0; j < data[i].size(); j++) {
			features(j, i) = data[i][j] / 255.0;
		}
	}

	return features;
}

void output_result(const Eigen::VectorXi& result, std::string filename) {
	std::ofstream outputFile(filename, std::ofstream::app);

	outputFile << "ImageId,Label" << std::endl;

	for (size_t i = 0; i < result.rows(); i++) {
		outputFile << i + 1 << "," << result(i) << std::endl;
	}

	outputFile.close();

}

std::string current_date_time() {
	auto now = std::chrono::system_clock::now();
	auto in_time_t = std::chrono::system_clock::to_time_t(now);

	std::stringstream ss;
	ss << std::put_time(std::localtime(&in_time_t), "%Y%m%d%H%M%S");
	return ss.str();
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

template <class Transformation, class Model, class TargetType>
double split_crossvalidation(Transformation& transformation, Model& model, const Eigen::MatrixXd& features, const TargetType& target, double training_percentage) {
	assert(features.cols() == target.cols() || (target.cols() == 1 && features.cols() == target.rows()));
	assert(training_percentage > 0 && training_percentage < 1);
	auto training = static_cast<size_t>(std::round(features.cols() * training_percentage));
	auto validation = features.cols() - training;
	assert(training > 0 && validation > 0);

	auto training_features = features.leftCols(training);
	auto training_target = target.cols() == 1 ? target.block(0, 0, training, 1) : target.block(0, 0, target.rows(), training);
	auto validation_features = features.rightCols(validation);
	auto validation_target = target.cols() == 1 ? target.block(training, 0, validation, 1) : target.block(0, training, target.rows(), validation);

	transformation.fit(training_features, training_target);
	model.fit(transformation.transform(training_features), training_target);

	return model.score(transformation.transform(validation_features), validation_target);
}

int main() {
	print_info();
	std::cout << std::endl;

	Eigen::MatrixXd features;
	Eigen::VectorXi classes;

	std::tie(features, classes) = load_training_data("train.csv");

	auto output_filename = "digit_recognizer_cross_val_" + current_date_time() + ".csv";
	std::ofstream output_file(output_filename);

	output_file << "Type;Batch Size;Learning Rate;Weight Decay;L2 Regularization;Score" << std::endl;

	for (auto batch_size : { 512, 256, 1024, 2048 }) {
		for (auto learning_rate : { 0.1, 0.01, 0.001, 0.0001 }) {
			for (auto decay : { 0.99, 0.95 }) {
				for (auto regularization : { 0.005, 0.0005 }) {

					using loss_t = mlt::utils::loss_functions::SoftmaxLoss;
					using act_t = mlt::utils::activation_functions::SigmoidActivation;
					using opt_t = mlt::utils::optimizers::StochasticGradientDescent<>;

					loss_t loss;
					act_t act;
					opt_t opt1(batch_size, 10, 0.001, decay);
					opt_t opt2(batch_size, 200, learning_rate, decay);

					auto model1 = mlt::models::transformers::create_sparse_autoencoder(196, act, act, opt1, 3e-3, 0.1, 3);
					mlt::models::classifiers::OptimizableLinearClassifier<loss_t, opt_t> model2(loss, opt2, regularization, true);

					double score = split_crossvalidation(model1, model2, features, classes, 0.8);
					std::cout << "Score for SparseAutoencoder -> Softmax " << batch_size << " " << learning_rate << " " << decay << " " << regularization << ": " << score << std::endl;
					output_file << "SparseAutoencoder -> Softmax;" << batch_size << ";" << learning_rate << ";" << decay << ";" << regularization << ";" << score << std::endl;

					double score2 = split_crossvalidation(model2, features, classes, 0.8);
					std::cout << "Score for Softmax " << batch_size << " " << learning_rate << " " << decay << " " << regularization << ": " << score2 << std::endl;
					output_file << "Softmax;" << batch_size << ";" << learning_rate << ";" << decay << ";" << regularization << ";" << score2 << std::endl;
				}
			}
		}
	}

	std::cin.get();

	return 0;
}