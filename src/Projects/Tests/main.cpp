#define EIGEN_USE_MKL_ALL

#include <fstream>
#include <iostream>
#include <stdio.h>
#include <time.h>

#include <Eigen/Core>
#include <omp.h>

#include "../../Components/LinearClassifiers/SoftmaxLinearClassifier.h"
#include "../../Components/PrincipalComponentAnalysis/CovariancePCA.h"

using namespace Eigen;
using namespace std;

void Softmax(vector<vector<int> >& set, bool optimize = false);

void parseCsv(string file, bool skipFirstLine, vector<vector<int> >& result);
const string currentDateTime();

void printIteration(unsigned long iter, const Eigen::VectorXd& x, double value, const Eigen::VectorXd& gradient);

int main()
{
	cout << "#Threads: " << Eigen::nbThreads() << endl;
	cout << "SIMD Instruction Sets In Use: " << Eigen::SimdInstructionSetsInUse() << endl;	
	
#ifdef EIGEN_USE_MKL
	cout << "MKL Enabled. Version: " << INTEL_MKL_VERSION << endl;
#endif

	vector<vector<int> > set;

	cout << "Loading train set" << endl;

	parseCsv("KaggleDigitRecognizer-train.csv", true, set);

	cout << "Loaded train set" << endl;

	Softmax(set, false);

	cin.get();

	return 0;
}

void Softmax(vector<vector<int> >& set, bool optimize)
{
	size_t totalSize = set.size();
	size_t features = set[0].size() - 1;
	size_t iterations = 250;
	double lambda = 3e-005;	
			
	MatrixXd trainingSet(set.size(), features);
	VectorXi trainingLabels(set.size());		

	cout << "Moving train set to Eigen Matrix" << endl;

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

	CovariancePCA pca;
	pca.train(trainingSet);
	MatrixXd projected = pca.transform(trainingSet);

	cout << "Started trainining" << endl;
	
	for (int i = 0; i < 10; i++)
	{
		SoftmaxLinearClassifier cl(projected.cols(), 10, lambda);
		LBFGS searchStrategy(50);
		ObjectiveDelta stopStrategy(1e-7, iterations);

		double dtime = omp_get_wtime();

		cl.train(projected, trainingLabels, searchStrategy, stopStrategy);

		dtime = omp_get_wtime() - dtime;

		cout << "Training Time: " << dtime << "s" << endl;
	}
	
	return;

	/*cin.get();

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

	outputFile.close();*/
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