#define EIGEN_USE_MKL_ALL

#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <chrono>

#include <Eigen/Core>

#include "utils/optimizers/stochastic_gradient_descent.hpp"
#include "utils/loss_functions.hpp"
#include "utils/activation_functions.hpp"
#include "../misc.hpp"
#include "models/transformers/sparse_autoencoder.hpp"
#include "models/classifiers/optimizable_linear_classifier.hpp"

using namespace std;

using namespace Eigen;

using namespace mlt::models::transformers;
using namespace mlt::models::classifiers;
using namespace mlt::utils;

vector<vector<double>> parseCsv(string file, bool skipFirstLine, char delim = ',') {
	vector<vector<double>> result;

	ifstream str(file.c_str());
	string line;

	if (skipFirstLine) {
		getline(str, line);
	}

	while (getline(str, line)) {
		vector<double> currentLine;
		stringstream lineStream(line);
		string cell;

		while (getline(lineStream, cell, delim)) {
			currentLine.push_back(stod(cell));
		}

		result.push_back(currentLine);
	}

	return result;
}

tuple<MatrixXd, VectorXi> load_training_data(string file) {
	auto data = parseCsv(file, true);
	MatrixXd features(data[0].size(), data.size());
	VectorXi classes(data.size());

	for(auto i = 0; i < data.size(); i++) {
		classes(i) = data[i][0];
		for (auto j = 1; j < data[i].size(); j++) {
			features(j - 1, i) = data[i][j] / 255.0;
		}
	}

	return{ features, classes };
}

MatrixXd load_test_data(string file) {
	auto data = parseCsv(file, true);
	MatrixXd features(data[0].size(), data.size());

	for (auto i = 0; i < data.size(); i++) {
		for (auto j = 0; j < data[i].size(); j++) {
			features(j, i) = data[i][j] / 255.0;
		}
	}

	return features;
}

void output_result(const VectorXi& result, string filename) {
	ofstream outputFile(filename, ofstream::app);

	outputFile << "ImageId,Label" << endl;

	for (size_t i = 0; i < result.rows(); i++) {
		outputFile << i + 1 << "," << result(i) << endl;
	}

	outputFile.close();

}

string current_date_time() {
	auto now = chrono::system_clock::now();
	auto in_time_t = chrono::system_clock::to_time_t(now);

	stringstream ss;
	ss << put_time(localtime(&in_time_t), "%Y%m%d%H%M%S");
	return ss.str();
}

template <class Model, class TargetType>
double split_crossvalidation(Model& model, const MatrixXd& features, const TargetType& target, double training_percentage) {
	assert(features.cols() == target.cols() || (target.cols() == 1 && features.cols() == target.rows()));
	assert(training_percentage > 0 && training_percentage < 1);
	auto training = static_cast<size_t>(round(features.cols() * training_percentage));
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
double split_crossvalidation(Transformation& transformation, Model& model, const MatrixXd& features, const TargetType& target, double training_percentage) {
	assert(features.cols() == target.cols() || (target.cols() == 1 && features.cols() == target.rows()));
	assert(training_percentage > 0 && training_percentage < 1);
	auto training = static_cast<size_t>(round(features.cols() * training_percentage));
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
	cout << endl;

	MatrixXd features;
	VectorXi classes;

	tie(features, classes) = load_training_data("train.csv");

	auto output_filename = "digit_recognizer_cross_val_" + current_date_time() + ".csv";
	ofstream output_file(output_filename);

	output_file << "Type;Batch Size;Learning Rate;Weight Decay;L2 Regularization;Score" << endl;

	for (auto batch_size : { 512, 256, 1024, 2048 }) {
		for (auto learning_rate : { 0.1, 0.01, 0.001, 0.0001 }) {
			for (auto decay : { 0.99, 0.95 }) {
				for (auto regularization : { 0.005, 0.0005 }) {

					using loss_t = loss_functions::SoftmaxLoss;
					using act_t = activation_functions::SigmoidActivation;
					using opt_t = optimizers::StochasticGradientDescent<>;

					loss_t loss;
					act_t act;
					opt_t opt1(batch_size, 10, 0.001, decay);
					opt_t opt2(batch_size, 200, learning_rate, decay);

					auto model1 = create_sparse_autoencoder(196, act, act, opt1, 3e-3, 0.1, 3);
					auto model2 = OptimizableLinearClassifier<loss_t, opt_t>(loss, opt2, regularization, true);

					auto score = split_crossvalidation(model1, model2, features, classes, 0.8);
					cout << "Score for SparseAutoencoder -> Softmax " << batch_size << " " << learning_rate << " " << decay << " " << regularization << ": " << score << endl;
					output_file << "SparseAutoencoder -> Softmax;" << batch_size << ";" << learning_rate << ";" << decay << ";" << regularization << ";" << score << endl;

					auto score2 = split_crossvalidation(model2, features, classes, 0.8);
					cout << "Score for Softmax " << batch_size << " " << learning_rate << " " << decay << " " << regularization << ": " << score2 << endl;
					output_file << "Softmax;" << batch_size << ";" << learning_rate << ";" << decay << ";" << regularization << ";" << score2 << endl;
				}
			}
		}
	}

	cin.get();

	return 0;
}