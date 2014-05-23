#include <fstream>
#include <iostream>
#include <time.h>

#include <Eigen/Core>
#include <omp.h>

#include "../../Components/FeedForwardNeuralNetwork/FeedForwardNeuralNetwork.h"
//#include "../../Components/PrincipalComponentAnalysis/PrincipalComponentAnalysis.h"

using namespace Eigen;
using namespace std;

void saveTheta(vector<MatrixXd>& theta, const char* file);
void loadTheta(vector<MatrixXd>& theta, const char* file);

void parseCsv(string file, bool skipFirstLine, vector<vector<string> >& result);
const string currentDateTime();

void printIteration(unsigned long iter, const Eigen::VectorXd& x, double cost, const Eigen::VectorXd& gradient);
bool fileExists(const string& name);

int main()
{
	cout << "#Threads: " << Eigen::nbThreads() << endl;
	cout << "SIMD Instruction Sets In Use: " << Eigen::SimdInstructionSetsInUse() << endl;

	vector<vector<string> > set;

	cout << "Loading Training Set..." << endl;

	parseCsv("training.csv", true, set);

	cout << "Loaded Training Set" << endl;

	size_t totalSize = set.size();
	size_t features = set[0].size() - 2;

	cout << "Size: " << totalSize << endl;
	cout << "Features: " << features << endl;

	int hiddenLayer = 500;
	size_t iterations = 2000;
	double lambda = 1;

	vector<size_t> layers;
	layers.push_back(hiddenLayer);
		
	MatrixXd trainingSet(totalSize, features);
	VectorXi trainingLabels(totalSize);

	for(unsigned int i = 0; i < totalSize ; i++)
	{		
		for(unsigned int j = 0; j < features; j++)
		{
			trainingSet(i, j) = atof(set[i][j].c_str());
		}

		trainingLabels(i) = set[i][features + 1] == "b";
	}

	vector<vector<string> >().swap(set);
	
	RowVectorXd means = trainingSet.colwise().mean();
	RowVectorXd stdDev = trainingSet.colwise().maxCoeff() - trainingSet.colwise().minCoeff();
	
	trainingSet.rowwise() -= means;
	trainingSet.array().rowwise() /= stdDev.array();

	vector<MatrixXd> theta(2);

	if(!fileExists("lastTraining.bin"))
	{
		FeedForwardNeuralNetwork nn(features, layers, 2);
		LBFGS searchStrategy(50);
		ObjectiveDelta stopStrategy(1e-7, iterations);

		cout << "Training Neural Network..." << endl;

		double dtime = omp_get_wtime();

		nn.train(trainingSet, trainingLabels, searchStrategy, stopStrategy.verbose(printIteration), lambda);

		dtime = omp_get_wtime() - dtime;

		theta = nn.getTheta();

		cout << "Neural Network Trained" << endl;

		cout << "Training Time: " << dtime << "s" << endl;
	}
	else
	{
		cout << "Loading Previously Trained Parameters..." << endl;

		loadTheta(theta, "lastTraining.bin");

		cout << "Previously Trained Parameters Loaded" << endl;
	}
	
	cout << "Loading Test Set..." << endl;

	parseCsv("test.csv", true, set);

	cout << "Loaded Test Set" << endl;

	MatrixXd testSet(set.size(), features);

	for (size_t i = 0; i < set.size(); i++)
	{
		for (size_t j = 0; j < set[i].size(); j++)
		{
			testSet(i, j) = atof(set[i][j].c_str());
		}
	}

	vector<vector<string> >().swap(set);

	testSet.rowwise() -= means;
	testSet.array().rowwise() /= stdDev.array();

	FeedForwardNeuralNetwork nn(features, layers, 2);

	cout << "Predicting Labels..." << endl;

	VectorXi predictions = nn.predictMany(testSet);

	cout << "Labels Predicted" << endl;

	stringstream ss;

	ss << "output" << "-";

	for (size_t l = 0; l < layers.size() - 1; l++)
	{
		ss << layers[l] << ".";
	}
	
	ss << layers[layers.size() - 1] << "-" << lambda << "-" << iterations << ".out";

	ofstream outputFile(ss.str().c_str(), std::ofstream::app);
	
	outputFile << "EventId,RankOrder,Class" << endl;
	
	for (size_t i = 0; i < predictions.rows(); i++)
	{
		outputFile << i + 350000 << "," << i + 1 << "," << (predictions(i) == 1 ? 'b' : 's') << endl;
	}

	outputFile.close();

	cin.get();

	return 0;
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

void parseCsv(string file, bool skipFirstLine, vector<vector<string> >& result)
{
	result.clear();
	ifstream str(file.c_str());
    string line;

	if(skipFirstLine)
	{
		getline(str, line);
	}

	int i = 0;

	while (getline(str, line))
	{
		vector<string> currentLine;
		std::stringstream lineStream(line);
		std::string cell;

		getline(lineStream, cell, ',');

		while(getline(lineStream, cell, ','))
		{
			currentLine.push_back(cell);
		}

		result.push_back(currentLine);

		i++;
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

void printIteration(unsigned long iter, const Eigen::VectorXd& x, double cost, const Eigen::VectorXd& gradient)
{
	std::cout << "iteration: " << iter << "   objective: " << cost << std::endl;
}

inline bool fileExists(const std::string& name) 
{
    ifstream f(name.c_str());

    if (f.good()) 
	{
        f.close();
        return true;
    }
	else 
	{
        f.close();
        return false;
    }   
}