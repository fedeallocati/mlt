#include <fstream>
#include <iostream>
#include <time.h>

#include <Eigen/Core>
#include <omp.h>

#include "../../Components/FeedForwardNeuralNetwork/Layer.h"
#include "../../Components/LinearClassifiers/SoftmaxClassifier.h"
#include "../../Components/FeedForwardNeuralNetwork/FeedForwardNeuralNetwork.h"
#include "../../Components/PrincipalComponentAnalysis/PrincipalComponentAnalysis.h"

using namespace Eigen;
using namespace std;

void Softmax(vector<vector<int> >& set, bool optimize = false);
void NN(vector<vector<int> >& set, bool optimize = false);

void saveTheta(vector<MatrixXd>& theta, const char* file);
void loadTheta(vector<MatrixXd>& theta, const char* file);

void parseCsv(string file, bool skipFirstLine, vector<vector<int> >& result);
const string currentDateTime();

void printIteration(unsigned long iter, const Eigen::VectorXd& x, double value, const Eigen::VectorXd& gradient);

int main()
{
	cout << "#Threads: " << Eigen::nbThreads() << endl;
	cout << "SIMD Instruction Sets In Use: " << Eigen::SimdInstructionSetsInUse() << endl;	

	vector<vector<int> > set;
	parseCsv("KaggleDigitRecognizer-train.csv", true, set);

	Softmax(set, true);

	cin.get();

	return 0;
}

void Softmax(vector<vector<int> >& set, bool optimize)
{
	size_t totalSize = set.size();
	size_t features = set[0].size() - 1;	
	size_t iterations = 250;
	double lambda = 1;

	if(optimize)
	{
		size_t trainingSize = set.size() * 3 / 4;
		size_t crossValSize = set.size() - trainingSize;

		MatrixXd trainingSet(trainingSize, features);
		VectorXi trainingLabels(trainingSize);

		MatrixXd crossValSet(crossValSize, features);
		VectorXi crossValLabels(crossValSize);

		for(unsigned int i = 0; i < trainingSize ; i++)
		{
			trainingLabels(i) = set[i][0];

			for(unsigned int j = 1; j < set[i].size(); j++)
			{
				trainingSet(i, j - 1) = set[i][j] - 128;
			}
		}

		for(unsigned int i = 0; i < crossValSize; i++)
		{
			crossValLabels(i) = set[i + trainingSize][0];

			for(unsigned int j = 1; j < set[i + trainingSize].size(); j++)
			{
				crossValSet(i, j - 1) = set[i + trainingSize][j] - 128;
			}
		}

		double maxVal = 255;

		trainingSet = trainingSet / maxVal;
		crossValSet = crossValSet / maxVal;

		PrincipalComponentAnalysis pca(trainingSet);

		MatrixXd projectedTrainingSet = pca.projectData(trainingSet);
		MatrixXd projectedCrossValSet = pca.projectData(crossValSet);

		double lambdas[] = { 1e-7, 3e-7, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3 };
		int lambdasSize = sizeof(lambdas) / sizeof(double);
		
		double maxHit = -1;
		lambda = -1;

		string file = "softmax-optimization-" + currentDateTime() + ".out";
				
		for(int i = 0; i < lambdasSize; i++)
		{								
			SoftmaxClassifier cl(projectedTrainingSet.cols(), 10);

			LBFGS searchStrategy(100);
			ObjectiveDelta stopStrategy(1e-9, iterations);

			cl.train(projectedTrainingSet, trainingLabels, searchStrategy, stopStrategy.verbose(printIteration), lambdas[i]);

			size_t hit = cl.predict(projectedCrossValSet).cwiseEqual(crossValLabels).count();

			ofstream optimizeFile(file.c_str(), std::ofstream::app);
						
			optimizeFile << lambdas[i] << ";" << hit << endl;

			optimizeFile.close();

			if(hit > maxHit)
			{
				maxHit = hit;					
				lambda = lambdas[i];
			}
		}

		cout << "Best Lambda: " << lambda << " - Accuracy: " << (double)maxHit / crossValSize << endl;		
	}
			
	MatrixXd trainingSet(set.size(), features);
	VectorXi trainingLabels(set.size());

	for(unsigned int i = 0; i < set.size() ; i++)
	{
		trainingLabels(i) = set[i][0];

		for(unsigned int j = 1; j < set[i].size(); j++)
		{
			trainingSet(i, j - 1) = set[i][j] - 128;
		}
	}

	double maxVal = trainingSet.maxCoeff();
	
	trainingSet = trainingSet / maxVal;

	PrincipalComponentAnalysis pca(trainingSet);

	MatrixXd projected = pca.projectData(trainingSet);
		
	SoftmaxClassifier cl(projected.cols(), 10);
	LBFGS searchStrategy(50);
	ObjectiveDelta stopStrategy(1e-7, iterations);

	double dtime = omp_get_wtime();

	cl.train(projected, trainingLabels, searchStrategy, stopStrategy.verbose(printIteration), lambda);

	dtime = omp_get_wtime() - dtime;

	cout << "Training Time: " << dtime << "s" << endl;
	
	cin.get();

	parseCsv("KaggleDigitRecognizer-test.csv", true, set);

	MatrixXd testSet(set.size(), features);

	for (size_t i = 0; i < set.size(); i++)
	{
		for (size_t j = 0; j < set[i].size(); j++)
		{
			testSet(i, j) = set[i][j] - 128;
		}
	}

	testSet = testSet / maxVal;

	VectorXi predictions = cl.predict(pca.projectData(testSet));

	stringstream ss;

	ss << "softmax-output" << "-";	
	
	ss << lambda << "-" << iterations << ".out";

	ofstream outputFile(ss.str().c_str(), std::ofstream::app);
	
	outputFile << "ImageId,Label" << endl;

	for (size_t i = 0; i < predictions.rows(); i++)
	{
		outputFile << i + 1 << "," << predictions(i) << endl;
	}

	outputFile.close();
}

void SoftmaxTests()
{
	MatrixXd theta(3, 5);
	theta << 0.0, 0.01, -0.05, 0.1, 0.05,
		     0.2, 0.7, 0.2, 0.05, 0.16,
			 -0.3, 0.0, -0.45, -0.2, 0.03;

	SoftmaxClassifier c(theta);

	MatrixXd x(2, 5);
	x << 1, -15, 22, -44, 56,
		 1, -15, 22, -44, 56;

	MatrixXd y(3, 2);
	y << 0, 0,
		 1, 1,
		 0, 0;

	//c.gradientInternal(theta, x, y, 0);

	cout << c.predict(x);
	
	cin.get();
}

void NN(vector<vector<int> >& set, bool optimize)
{
	size_t totalSize = set.size();
	size_t features = set[0].size() - 1;
	int hiddenLayer = 500;
	size_t iterations = 250;
	double lambda = 1;

	if(optimize)
	{
		size_t trainingSize = set.size() * 3 / 4;
		size_t crossValSize = set.size() - trainingSize;

		MatrixXd trainingSet(trainingSize, features);
		VectorXi trainingLabels(trainingSize);

		MatrixXd crossValSet(crossValSize, features);
		VectorXi crossValLabels(crossValSize);

		for(unsigned int i = 0; i < trainingSize ; i++)
		{
			trainingLabels(i) = set[i][0];

			for(unsigned int j = 1; j < set[i].size(); j++)
			{
				trainingSet(i, j - 1) = set[i][j] - 128;
			}
		}

		for(unsigned int i = 0; i < crossValSize; i++)
		{
			crossValLabels(i) = set[i + trainingSize][0];

			for(unsigned int j = 1; j < set[i + trainingSize].size(); j++)
			{
				crossValSet(i, j - 1) = set[i + trainingSize][j] - 128;
			}
		}

		double maxVal = 255;

		trainingSet = trainingSet / maxVal;
		crossValSet = crossValSet / maxVal;

		PrincipalComponentAnalysis pca(trainingSet);

		MatrixXd projectedTrainingSet = pca.projectData(trainingSet);
		MatrixXd projectedCrossValSet = pca.projectData(crossValSet);

		double lambdas[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
		int hiddenLayers[] = {250, 500, 1000};
		int lambdasSize = sizeof(lambdas) / sizeof(double);
		int hiddenLayersSize = sizeof(hiddenLayers) / sizeof(int);

		double maxHit = -1;
		hiddenLayer = -1;
		lambda = -1;

		string file = "optimization-" + currentDateTime() + ".out";

		for(int i = 0; i < hiddenLayersSize; i++)
		{
			for(int j = 0; j < lambdasSize; j++)
			{
				vector<size_t> layers;
				layers.push_back(hiddenLayers[i]);				
				FeedForwardNeuralNetwork nn(projectedTrainingSet.cols(), layers, 10);

				LBFGS searchStrategy(50);
				ObjectiveDelta stopStrategy(1e-7, iterations);

				nn.train(projectedTrainingSet, trainingLabels, searchStrategy, stopStrategy.verbose(printIteration), lambdas[j]);

				size_t hit = nn.predictMany(projectedCrossValSet).cwiseEqual(crossValLabels).count();

				ofstream optimizeFile(file.c_str(), std::ofstream::app);

				for (size_t l = 0; l < layers.size() - 1; l++)
				{
					optimizeFile << layers[l] << "-";
				}

				optimizeFile << hiddenLayers[i] << ";" << lambdas[j] << ";" << hit << endl;

				optimizeFile.close();

				if(hit > maxHit)
				{
					maxHit = hit;
					hiddenLayer = hiddenLayers[i];
					lambda = lambdas[j];
				}
			}
		}
	}

	vector<size_t> layers;
	layers.push_back(hiddenLayer);
		
	MatrixXd trainingSet(set.size(), features);
	VectorXi trainingLabels(set.size());

	for(unsigned int i = 0; i < set.size() ; i++)
	{
		trainingLabels(i) = set[i][0];

		for(unsigned int j = 1; j < set[i].size(); j++)
		{
			trainingSet(i, j - 1) = set[i][j] - 128;
		}
	}

	double maxVal = trainingSet.maxCoeff();
	
	trainingSet = trainingSet / maxVal;

	PrincipalComponentAnalysis pca(trainingSet);

	MatrixXd projected = pca.projectData(trainingSet);
		
	FeedForwardNeuralNetwork nn(projected.cols(), layers, 10);
	LBFGS searchStrategy(50);
	ObjectiveDelta stopStrategy(1e-7, iterations);

	double dtime = omp_get_wtime();

	nn.train(projected, trainingLabels, searchStrategy, stopStrategy.verbose(printIteration), lambda);

	dtime = omp_get_wtime() - dtime;

	cout << "Training Time: " << dtime << "s" << endl;
	
	cin.get();

	parseCsv("KaggleDigitRecognizer-test.csv", true, set);

	MatrixXd testSet(set.size(), features);

	for (size_t i = 0; i < set.size(); i++)
	{
		for (size_t j = 0; j < set[i].size(); j++)
		{
			testSet(i, j) = set[i][j] - 128;
		}
	}

	testSet = testSet / maxVal;

	VectorXi predictions = nn.predictMany(pca.projectData(testSet));

	stringstream ss;

	ss << "output" << "-";

	for (size_t l = 0; l < layers.size() - 1; l++)
	{
		ss << layers[l] << ".";
	}
	
	ss << layers[layers.size() - 1] << "-" << lambda << "-" << iterations << ".out";

	ofstream outputFile(ss.str().c_str(), std::ofstream::app);
	
	outputFile << "ImageId,Label" << endl;

	for (size_t i = 0; i < predictions.rows(); i++)
	{
		outputFile << i + 1 << "," << predictions(i) << endl;
	}

	outputFile.close();
}

void saveTheta(vector<MatrixXd>& theta, const char* file)
{
	std::ofstream f(file, std::ios::binary);

	for (size_t i = 0; i < theta.size(); i++)
	{
		Eigen::MatrixXd::Index rows, cols;
		rows = theta[i].rows();
		cols = theta[i].cols();

		f.write((char *)&rows, sizeof(rows));
		f.write((char *)&cols, sizeof(cols));
		f.write((char *)theta[i].data(), sizeof(Eigen::MatrixXd::Scalar) * rows * cols);
	}

	f.close();
}

void loadTheta(vector<MatrixXd>& theta, const char* file)
{
	std::ifstream f(file, std::ios::binary);

	for(unsigned int i = 0; i < theta.size(); i++)
	{
		Eigen::MatrixXd::Index rows, cols;
	
		f.read((char *)&rows, sizeof(rows));
		f.read((char *)&cols, sizeof(cols));

		theta[i].resize(rows, cols);

		f.read((char *)theta[i].data(), sizeof(Eigen::MatrixXd::Scalar) * rows * cols);

		if (f.bad())
		{
			throw "Error reading matrix";
		}
	}

	f.close();
}

void parseCsv(string file, bool skipFirstLine, vector<vector<int> >& result)
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
		vector<int> currentLine;
		std::stringstream lineStream(line);
		std::string cell;

		while(getline(lineStream, cell, ','))
		{
			currentLine.push_back(atoi(cell.c_str()));
		}

		result.push_back(currentLine);
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

void printIteration(unsigned long iter, const Eigen::VectorXd& x, double value, const Eigen::VectorXd& gradient)
{
	std::cout << "iteration: " << iter << "   objective: " << value << std::endl;
}
