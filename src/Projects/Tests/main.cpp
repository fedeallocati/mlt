#define EIGEN_VECTORIZE_SSE4_2
#define EIGEN_USE_MKL_ALL

#include <fstream>
#include <iostream>
#include <stdio.h>

#include <Eigen/Core>
#include <omp.h>

#include "../../mlt/linear_classifiers/softmax_linear_classifier.h"
#include "../../mlt/linear_classifiers/svm_linear_classifier.h"
#include "../../mlt/trainers/gradient_descent/gradient_descent_trainer.h"
#include "../../mlt/dimensionality_reduction/principal_component_analysis.h"

using namespace std;
using namespace Eigen;
using namespace MLT::LinearClassifiers;
using namespace MLT::Trainers::GradientDescent;
using namespace MLT::DimensionalityReduction;

void run(GradientDescentTrainableLinearClassifier& cl);

void classify(IClassifier& cl, double normalizationFactor, PrincipalComponentAnalysis& trainedPca);

void parse_csv(string file, bool skipFirstLine, vector<vector<int> >& result);

int main()
{
	cout << "#Threads: " << Eigen::nbThreads() << endl;
	cout << "SIMD Instruction Sets In Use: " << Eigen::SimdInstructionSetsInUse() << endl;	
	
#ifdef EIGEN_USE_MKL
	cout << "MKL Enabled. Version: " << INTEL_MKL_VERSION << endl;
#endif	

	if (false)
	{
		SoftmaxLinearClassifier cl(196, 10, 0.001, 3e-005);
		run(cl);
	}
	else
	{
		SvmLinearClassifier cl(196, 10, 0.001, 3e-005);
		run(cl);
	}

	cin.get();

	return 0;
}

void run(GradientDescentTrainableLinearClassifier& cl)
{
	vector<vector<int> > training_set;
	cout << "Loading training set" << endl;
	parse_csv("KaggleDigitRecognizer-train.csv", true, training_set);

	size_t m = training_set.size();
	size_t features = training_set[0].size() - 1;

	MatrixXd training_set_eigen(m, features);
	MatrixXd training_labels_eigen = MatrixXd::Zero(10, m);

	cout << "Moving training set to Eigen Matrix" << endl;

	for (unsigned int i = 0; i < m; i++)
	{
		training_labels_eigen(training_set[i][0], i) = 1;

		for (unsigned int j = 1; j < features; j++)
		{
			training_set_eigen(i, j - 1) = training_set[i][j] - 128;
		}
	}

	double maxVal = training_set_eigen.maxCoeff();
	training_set_eigen = training_set_eigen / maxVal;

	cout << "Computing PCA" << endl;
	PrincipalComponentAnalysis pca;
	pca.train(training_set_eigen);
	training_set_eigen = pca.transform(training_set_eigen);

	cout << "Started trainining" << endl;

	GradientDescentTrainer trainer;

	trainer.train(cl, training_set_eigen, training_labels_eigen);

	classify(cl, maxVal, pca);
}

void classify(IClassifier& cl, double normalization, PrincipalComponentAnalysis& pca)
{
	vector<vector<int> > test_set;
	cout << "Loading test set" << endl;
	parse_csv("KaggleDigitRecognizer-test.csv", true, test_set);

	cout << "Moving test set to Eigen Matrix" << endl;

	MatrixXd test_set_eigen(test_set.size(), test_set[0].size());

	for (size_t i = 0; i < test_set.size(); i++)
	{
		for (size_t j = 0; j < test_set[i].size(); j++)
		{
			test_set_eigen(i, j) = test_set[i][j] - 128;
		}
	}

	test_set_eigen = test_set_eigen / normalization;
	test_set_eigen = pca.transform(test_set_eigen);

	cout << "Started classifing" << endl;

	VectorXi predictions = cl.classify(test_set_eigen);

	stringstream ss;

	ofstream output_file("prediction.out", std::ofstream::app);

	output_file << "ImageId,Label" << endl;

	for (size_t i = 0; i < predictions.rows(); i++)
	{
		output_file << i + 1 << "," << predictions(i) << endl;
	}

	output_file.close();
}


void parse_csv(string file, bool skipFirstLine, vector<vector<int> >& result)
{
	result.clear();
	ifstream str(file.c_str());
    string line;

	if(skipFirstLine)
	{
		getline(str, line);
	}

	while (getline(str, line))
	{
		vector<int> current_line;
		std::stringstream line_stream(line);
		std::string cell;

		while (getline(line_stream, cell, ','))
		{
			current_line.push_back(atoi(cell.c_str()));
		}

		result.push_back(current_line);
	}
}