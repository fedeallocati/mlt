#define EIGEN_VECTORIZE_SSE4_2
#define EIGEN_USE_MKL_ALL

#include <fstream>
#include <iostream>
#include <time.h>
#include <Eigen/Core>
#include <omp.h>

#include "../../mlt/linear_classifiers/softmax_linear_classifier.h"
#include "../../mlt/trainers/gradient_descent/gradient_descent_trainer.h"
#include "../../mlt/neural_networks/multilayer_perceptron_classifier.h"
#include "../../mlt/dimensionality_reduction/principal_component_analysis.h"

using namespace std;
using namespace Eigen;
using namespace MLT::LinearClassifiers;
using namespace MLT::NeuralNetworks;
using namespace MLT::Trainers::GradientDescent;
using namespace MLT::DimensionalityReduction;

void NN(bool optimize);
void SoftmaxLinear(bool optimize);

void parseCsv(string file, bool skipFirstLine, vector<vector<int> >& result, bool verbose = false);
const string currentDateTime();

bool askYesNoQuestion(const string& question)
{
	while (true)
	{
		cout << question << " ";
		string response;
		cin >> response;

		if (response == "N" || response == "n" || response == "No" || response == "no" || response == "NO")
		{
			cout << endl;
			return false;
		}

		if (response == "Y" || response == "y" || response == "Yes" || response == "yes" || response == "YES")
		{
			cout << endl;
			return true;
		}

		cout << "Invalid answer" << endl;
	}
}

void printIteration(unsigned long iter, const Eigen::VectorXd& x, double loss, const Eigen::VectorXd& gradient);

void buildInfo()
{
#ifdef EIGEN_USE_MKL
	cout << "Using MKL " << INTEL_MKL_VERSION << " as backend" << endl;
	cout << "Max Threads: " << MKL_Get_Max_Threads() << endl;
	cout << "SIMD Instruction Sets In Use: " << Eigen::SimdInstructionSetsInUse() << endl;
#else
	cout << "Using Eigen " << EIGEN_WORLD_VERSION << "." << EIGEN_MAJOR_VERSION << "." << EIGEN_MINOR_VERSION << " as backend" << endl;
	cout << "Max Threads: " << Eigen::nbThreads() << endl;
	cout << "SIMD Instruction Sets In Use: " << Eigen::SimdInstructionSetsInUse() << endl;
#endif	
}

int main()
{	
	buildInfo();
	cout << endl;

	bool mlp = askYesNoQuestion("Use Multilayer Perceptron?");
	bool optimize = askYesNoQuestion("Optimize Hyperparameters?");

	if (mlp)
	{
		NN(optimize);
	}
	else
	{
		SoftmaxLinear(optimize);
	}
	
	cin.get();

	return 0;
}

void get_training_set(MatrixXd& training_data, VectorXi& training_labels, MatrixXd& training_target)
{
	vector<vector<int> > set;

	cout << "Loading training set" << endl;
	double dtime = omp_get_wtime();

	parseCsv("KaggleDigitRecognizer-train.csv", true, set);

	dtime = omp_get_wtime() - dtime;
	cout << "Load time: " << dtime << "s" << endl;
		
	cout << "Moving training set to Eigen" << endl ;
	dtime = omp_get_wtime();

	size_t total_size = set.size();
	size_t features = set[0].size() - 1;

	training_data = MatrixXd(total_size, features);
	training_labels = VectorXi(total_size);
	training_target = MatrixXd::Zero(10, total_size);

	for (unsigned int i = 0; i < total_size; i++)
	{
		training_labels(i) = set[i][0];
		training_target(set[i][0], i) = 1;

		for (unsigned int j = 1; j < features + 1; j++)
		{
			training_data(i, j - 1) = (set[i][j] - 128) / 128;
		}
	}

	dtime = omp_get_wtime() - dtime;
	cout << "Move time: " << dtime << "s" << endl << endl;
}

void get_test_set(MatrixXd& test_data)
{
	vector<vector<int> > set;

	cout << "Loading test set" << endl;
	double dtime = omp_get_wtime();

	parseCsv("KaggleDigitRecognizer-test.csv", true, set);

	dtime = omp_get_wtime() - dtime;
	cout << "Load time: " << dtime << "s" << endl;

	cout << "Moving test set to Eigen" << endl;
	dtime = omp_get_wtime();

	size_t total_size = set.size();
	size_t features = set[0].size();

	test_data = MatrixXd(total_size, features);

	for (unsigned int i = 0; i < total_size; i++)
	{
		for (unsigned int j = 1; j < features + 1; j++)
		{
			test_data(i, j - 1) = (set[i][j] - 128) / 128;
		}
	}

	dtime = omp_get_wtime() - dtime;
	cout << "Move time: " << dtime << "s" << endl << endl;
}

void output_result(const VectorXi& result, const string& filename)
{	
	cout << "Outputing to file " << filename << endl;

	ofstream outputFile(filename.c_str(), std::ofstream::app);

	outputFile << "ImageId,Label" << endl;

	for (size_t i = 0; i < result.rows(); i++)
	{
		outputFile << i + 1 << "," << result(i) << endl;
	}

	outputFile.close();

}

void run()
{

}

void NN(bool optimize)
{	
	int hidden_layer_1 = 500;
	int hidden_layer_2 = 0;	
	size_t iterations = 500;	
	double lambda = 1;

	MatrixXd training_data;
	VectorXi training_labels;
	MatrixXd training_target;

	get_training_set(training_data, training_labels, training_target);

	cout << "Training PCA" << endl;
	PrincipalComponentAnalysis pca;
	double dtime = omp_get_wtime();
	pca.train(training_data);
	dtime = omp_get_wtime() - dtime;
	cout << "Training time: " << dtime << "s" << endl << endl;

	training_data = pca.transform(training_data);

	GradientDescentTrainer trainer;

	if(optimize)
	{	
		size_t training_size = training_data.rows() * 3 / 4;
		size_t cross_val_size = training_data.rows() - training_size;

		double lambdas[] = { 1, 10, 20, 30 };
		int hidden_layers_1[] = { 250, 500, 1000 };
		int hidden_layers_2[] = { 0, 250, 500, 1000 };
		int lambdas_size = sizeof(lambdas) / sizeof(double);
		int hidden_layers_1_size = sizeof(hidden_layers_1) / sizeof(int);
		int hidden_layers_2_size = sizeof(hidden_layers_2) / sizeof(int);

		double max_hit = -1;
		hidden_layer_1 = -1;
		hidden_layer_2 = -1;
		lambda = -1;

		string file = "mlp-optimization-" + currentDateTime() + ".out";

		cout << "Choosing best hyperparameters with Cross-Validation" << endl << endl;

		for (int i = 0; i < hidden_layers_1_size; i++)
		{
			for (int j = 0; j < hidden_layers_2_size; j++)
			{
				for (int k = 0; k < lambdas_size; k++)
				{
					vector<size_t> layers;

					cout << "Training architecture: [" << pca.getNumberOfFeatures() << " - "
						<< hidden_layers_1[i] << " - " << hidden_layers_2[j] << " - "
						<< 10 << "] and lambda: " << lambdas[k] << endl;

					if (hidden_layers_1[i] != 0)
					{
						layers.push_back(hidden_layers_1[i]);
					}					

					if (hidden_layers_2[j] != 0)
					{
						layers.push_back(hidden_layers_2[j]);
					}
					
					MultilayerPerceptronClassifier nn(pca.getNumberOfFeatures(), layers, 10, 0.12, lambdas[k]);

					dtime = omp_get_wtime();

					trainer.train(nn, training_data.topRows(training_size), training_target.leftCols(training_size));

					dtime = omp_get_wtime() - dtime;

					cout << "Training time: " << dtime << "s" << endl;

					size_t train_hit = nn.classify(training_data.topRows(training_size)).cwiseEqual(training_labels.topRows(training_size)).count();
					size_t cross_val_hit = nn.classify(training_data.bottomRows(cross_val_size)).cwiseEqual(training_labels.bottomRows(cross_val_size)).count();

					cout << "Traning set accuracy: " << train_hit / (double)training_size << ". Cross-Validation set accuracy: "
						<< cross_val_hit / (double)cross_val_size << endl << endl;

					ofstream optimizeFile(file.c_str(), std::ofstream::app);
					
					optimizeFile << hidden_layers_1[i] << "-" << hidden_layers_2[j] << ";" << lambdas[k] << ";" <<
						train_hit / (double)training_size << ";" << cross_val_hit / (double)cross_val_size << endl;

					optimizeFile.close();

					if (cross_val_hit > max_hit)
					{
						max_hit = cross_val_hit;
						hidden_layer_1 = hidden_layers_1[i];
						hidden_layer_2 = hidden_layers_2[j];
						lambda = lambdas[k];
					}
				}
			}
		}
		
		cout << "Best architecture: [" << pca.getNumberOfFeatures() << " - "
			<< hidden_layer_1 << " - " << hidden_layer_2 << " - "
			<< 10 << "] and lambda: " << lambda
			<< ", with accuracy: " << max_hit / (double)cross_val_size << endl << endl;
	}

	cout << "Training for evaluation with architecture: [" << pca.getNumberOfFeatures() << " - "
		<< hidden_layer_1 << " - " << hidden_layer_2 << " - "
		<< 10 << "] and lambda: " << lambda << endl;

	vector<size_t> layers;

	if (hidden_layer_1 != 0)
	{
		layers.push_back(hidden_layer_1);
	}

	if (hidden_layer_2 != 0)
	{
		layers.push_back(hidden_layer_2);
	}
		
	MultilayerPerceptronClassifier nn(pca.getNumberOfFeatures(), layers, 10, 0.12, lambda);
	
	dtime = omp_get_wtime();

	trainer.train(nn, training_data, training_target);

	dtime = omp_get_wtime() - dtime;

	cout << "Training time: " << dtime << "s" << endl << endl;

	cin.get();

	MatrixXd test_data;

	get_test_set(test_data);

	cout << "Doing predictions" << endl;

	VectorXi result = nn.classify(pca.transform(test_data));

	stringstream ss;

	ss << "mlp-output" << "-";

	for (size_t l = 0; l < layers.size() - 1; l++)
	{
		ss << layers[l] << "-";
	}
	
	ss << layers[layers.size() - 1] << "-" << lambda << "-" << iterations << ".out";

	output_result(result, ss.str());

	cout << "Finished" << endl;
}

void SoftmaxLinear(bool optimize)
{
	size_t iterations = 500;
	double lambda = 1;

	MatrixXd training_data;
	VectorXi training_labels;
	MatrixXd training_target;

	get_training_set(training_data, training_labels, training_target);

	cout << "Training PCA" << endl;
	PrincipalComponentAnalysis pca;
	double dtime = omp_get_wtime();
	pca.train(training_data);
	dtime = omp_get_wtime() - dtime;
	cout << "Training time: " << dtime << "s" << endl << endl;

	training_data = pca.transform(training_data);

	GradientDescentTrainer trainer;

	if (optimize)
	{
		size_t training_size = training_data.rows() * 3 / 4;
		size_t cross_val_size = training_data.rows() - training_size;		

		double lambdas[] = { 3e-5, 5e-5, 8e-5, 1e-4, 3e-4, 5e-4, 8e-4, 1e-3 ,3e-3 };
		int lambdas_size = sizeof(lambdas) / sizeof(double);	

		double max_hit = -1;		
		lambda = -1;

		string file = "softmax-optimization-" + currentDateTime() + ".out";

		cout << "Choosing best hyperparameters with Cross-Validation" << endl << endl;
				
		for (int i = 0; i < lambdas_size; i++)
		{
			cout << "Training with lambda " << lambdas[i] << endl;

			SoftmaxLinearClassifier cl(pca.getNumberOfFeatures(), 10, 0.001, lambdas[i]);
						
			dtime = omp_get_wtime();

			trainer.train(cl, training_data.topRows(training_size), training_target.leftCols(training_size));

			dtime = omp_get_wtime() - dtime;

			cout << "Training time: " << dtime << "s" << endl;
			
			size_t train_hit = cl.classify(training_data.topRows(training_size)).cwiseEqual(training_labels.topRows(training_size)).count();
			size_t cross_val_hit = cl.classify(training_data.bottomRows(cross_val_size)).cwiseEqual(training_labels.bottomRows(cross_val_size)).count();

			cout << "Traning set accuracy: " << train_hit / (double)training_size << ". Cross-Validation set accuracy: "
				<< cross_val_hit / (double)cross_val_size << endl << endl;

			ofstream optimizeFile(file.c_str(), std::ofstream::app);
						
			optimizeFile << lambdas[i] << ";" << train_hit / (double)training_size << ";" << cross_val_hit / (double)cross_val_size << endl;

			optimizeFile.close();

			if (cross_val_hit > max_hit)
			{
				max_hit = cross_val_hit;
				lambda = lambdas[i];
			}
		}
			
		cout << "Best lambda: " << lambda << ", with accuracy: " << max_hit / (double)cross_val_size << endl << endl;
	}

	cout << "Training for evaluation with lambda " << lambda << endl;

	SoftmaxLinearClassifier cl(pca.getNumberOfFeatures(), 10, 0.001, lambda);
	
	dtime = omp_get_wtime();

	trainer.train(cl, training_data, training_target);

	dtime = omp_get_wtime() - dtime;

	cout << "Training time: " << dtime << "s" << endl << endl;

	cin.get();

	MatrixXd test_data;

	get_test_set(test_data);

	cout << "Doing predictions" << endl;
		
	VectorXi result = cl.classify(pca.transform(test_data));

	stringstream ss;
	ss << "softmax-output" << "-" << lambda << "-" << iterations << ".out";

	output_result(result, ss.str());
	
	cout << "Finished" << endl;
}

void parseCsv(string file, bool skipFirstLine, vector<vector<int> >& result, bool verbose)
{
	result.clear();
	ifstream str(file.c_str());
    string line;

	if(skipFirstLine)
	{
		if (verbose)
		{
			cout << "Skipped first line" << endl;
		}
		
		getline(str, line);
	}

	while (getline(str, line))
	{
		vector<int> currentLine;
		stringstream lineStream(line);
		string cell;

		while(getline(lineStream, cell, ','))
		{
			currentLine.push_back(atoi(cell.c_str()));
		}

		result.push_back(currentLine);

		if (verbose)
		{
			cout << "Loaded " << result.size() << " rows" << endl;
		}		
	}
}

const string currentDateTime()
{
	time_t     now = time(0);
    struct tm  tstruct;
    char       buf[80];
    tstruct = *localtime(&now);
    
    strftime(buf, sizeof(buf), "%Y%m%d%H%M%S", &tstruct);

    return buf;
}

void printIteration(unsigned long iter, const Eigen::VectorXd& x, double loss, const Eigen::VectorXd& gradient)
{
	if (iter % 10 == 0)
	{
		std::cout << "iteration: " << iter << "   loss: " << loss << std::endl;
	}	
}