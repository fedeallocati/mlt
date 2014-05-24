#include <fstream>
#include <iostream>
#include <time.h>

#include <Eigen/Core>
#include <omp.h>

#include "../../Components/FeedForwardNeuralNetwork/FeedForwardNeuralNetwork.h"
//#include "../../Components/PrincipalComponentAnalysis/PrincipalComponentAnalysis.h"

using namespace Eigen;
using namespace std;

bool askYesNoQuestion(const string& question);

bool fileExists(const string& file);

void saveTheta(const vector<MatrixXd>& theta);
void loadTheta(vector<MatrixXd>& theta);

void saveNormalizations(const RowVectorXd& means, RowVectorXd& stdDev);
void loadNormalizations(RowVectorXd& means, RowVectorXd& stdDev);

//void savePca(const PrincipalComponentAnalysis& pca);
//void loadPca(PrincipalComponentAnalysis& pca);

void parseCsv(const string& file, vector<vector<string> >& input);
const string currentDateTime();

void printIteration(unsigned long iter, const Eigen::VectorXd& x, double cost, const Eigen::VectorXd& gradient);

unsigned int trainedIterations = 0;
string thetaFile = "KaggleHiggsBoson-lastTrainedTheta.bin";
string normalizationFile = "KaggleHiggsBoson-lastNormalizations.bin";
//string pcaFile = "KaggleHiggsBoson-lastPca.bin";

int main()
{
	cout << "#Threads: " << Eigen::nbThreads() << endl;
	cout << "SIMD Instruction Sets In Use: " << Eigen::SimdInstructionSetsInUse() << endl << endl;

	int hiddenLayer = 100;
	size_t iterations = 5000;
	double lambda = 1;

	vector<size_t> layers;
	layers.push_back(hiddenLayer);	
	size_t features = 30;
	size_t labels = 2;

	cout << "Original Features: " << features << endl;

	cout << "Hidden Layers: ";
	for (size_t l = 0; l < layers.size() - 1; l++)
	{
		cout << layers[l] << " - ";
	}
	cout<< layers[layers.size() - 1] << endl;

	cout << "Output: " << labels << endl;
	cout << "Lambda: " << lambda << endl << endl;
		
	vector<MatrixXd> theta;
	RowVectorXd means;
	RowVectorXd stdDev;	
	//PrincipalComponentAnalysis pca;

	bool loadedData = false;

	if(fileExists(thetaFile) && fileExists(normalizationFile) /*&& fileExists(pcaFile)*/)
	{
		if(askYesNoQuestion("Previous training files found. Load them?"))
		{
			cout << "Loading Previously Trained Parameters..." << endl;

			loadTheta(theta);
			loadNormalizations(means, stdDev);
			//loadPca(pca);

			cout << "Previously Trained Parameters Loaded" << endl << endl;		

			loadedData = true;
		}
	}
		
	bool train = askYesNoQuestion("Train neural network?");
	bool predict = askYesNoQuestion("Predict labels?");

	if (train && !fileExists("KaggleHiggsBoson-training.csv"))
	{
		train = true;
		cout << "File KaggleHiggsBoson-training.csv not found, can't train" << endl;
	}

	if (train)
	{
		vector<vector<string> > set;

		cout << "Loading Training Set (KaggleHiggsBoson-training.csv)..." << endl;

		parseCsv("KaggleHiggsBoson-training.csv", set);

		cout << "Loaded Training Set" << endl;
	
		size_t totalSize = set.size() - 1;
		cout << "Size: " << totalSize << endl << endl;	
		
		MatrixXd trainingSet(totalSize, features);
		VectorXi trainingLabels(totalSize);

		for(unsigned int i = 0; i < totalSize; i++)
		{		
			for(unsigned int j = 0; j < features; j++)
			{
				trainingSet(i, j) = atof(set[i + 1][j + 1].c_str());
			}

			trainingLabels(i) = set[i + 1][features + 2] == "b";
		}

		vector<vector<string> >().swap(set);
		
		if (!loadedData)
		{
			means = trainingSet.colwise().mean();
			stdDev = trainingSet.colwise().maxCoeff() - trainingSet.colwise().minCoeff();
			
			/*for(size_t i = 0; i < trainingSet.rows(); i++)
			{
				trainingSet.row(i) = (trainingSet.row(i) - means).cwiseQuotient(stdDev);
			}*/
			
			saveNormalizations(means, stdDev);			
		}

		trainingSet = (trainingSet.rowwise() - means).array().rowwise() / stdDev.array();

		/*if (!loadedData)
		{
			pca.train(trainingSet);
			cout << "Post PCA Features: " << pca.getNumberOfFeatures() << endl;
			savePca(pca);
		}*/		

		FeedForwardNeuralNetwork nn(/*pca.getNumberOfFeatures()*/features, layers, 2);

		if (loadedData)
		{
			nn.setTheta(theta);
		}

		LBFGS searchStrategy(50);
		ObjectiveDelta stopStrategy(1e-7, iterations);
		nn.debug(&saveTheta);

		cout << "Training Neural Network..." << endl;

		double dtime = omp_get_wtime();

		nn.train(/*pca.projectData(*/trainingSet/*)*/, trainingLabels, searchStrategy, stopStrategy.verbose(printIteration), lambda);

		dtime = omp_get_wtime() - dtime;

		cout << "Neural Network Trained" << endl;

		cout << "Training Time: " << dtime << "s" << endl << endl;
	}

	if (predict && !loadedData && !train)
	{
		predict = false;
		cout << "Didn't loaded previous data and  didn't trained, can't predict labels" << endl;
	}

	if (predict && !fileExists("KaggleHiggsBoson-test.csv"))
	{
		predict = false;
		cout << "File KaggleHiggsBoson-test.csv not found, can't predict" << endl;
	}

	if (predict)
	{
		vector<vector<string> > set;

		cout << "Loading Test Set (KaggleHiggsBoson-test.csv)..." << endl;

		parseCsv("KaggleHiggsBoson-test.csv", set);

		cout << "Loaded Test Set" << endl;

		MatrixXd testSet(set.size() - 1, features);

		for (size_t i = 0; i < set.size() - 1; i++)
		{
			for (size_t j = 0; j < features; j++)
			{				
				testSet(i, j) = atof(set[i + 1][j + 1].c_str());
			}
		}

		vector<vector<string> >().swap(set);

		testSet = (testSet.rowwise() - means).array().rowwise() / stdDev.array();
	
		cout << "Predicting Labels..." << endl;

		FeedForwardNeuralNetwork nn(/*pca.getNumberOfFeatures()*/features, layers, 2, theta);

		VectorXi predictions = nn.predictMany(/*pca.projectData(*/testSet/*)*/);

		cout << "Labels Predicted" << endl;

		stringstream ss;

		ss << "KaggleHiggsBoson-output" << "-" << /*pca.getNumberOfFeatures()*/features << "-";

		for (size_t l = 0; l < layers.size() - 1; l++)
		{
			ss << layers[l] << "-";
		}
	
		ss << layers[layers.size() - 1] << "-" << lambda << "-" << trainedIterations << ".out";

		ofstream outputFile(ss.str().c_str(), std::ofstream::app);
	
		outputFile << "EventId,RankOrder,Class" << endl;
	
		for (size_t i = 0; i < predictions.rows(); i++)
		{
			outputFile << i + 350000 << "," << i + 1 << "," << (predictions(i) == 1 ? 'b' : 's') << endl;
		}

		outputFile.close();
	}

	cin.get();
	cin.get();	

	return 0;
}

bool askYesNoQuestion(const string& question)
{
	while(true)
	{
		cout << question << " ";
		string response;
		cin >> response;

		if(response  == "N" || response == "n" || response == "No" || response == "no" || response == "NO")
		{
			cout << endl;
			return false;
		}

		if(response == "Y" || response == "y" || response == "Yes" || response == "yes" || response == "YES")
		{
			cout << endl;
			return true;
		}

		cout << "Invalid answer" << endl;
	}
}

bool fileExists(const string& file) 
{
	ifstream f(file.c_str());

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

void saveTheta(const vector<MatrixXd>& theta)
{
	std::ofstream f(thetaFile.c_str(), std::ios::binary);

	size_t thetas = theta.size();
	f.write((char *)&thetas, sizeof(thetas));

	f.write((char *)&trainedIterations, sizeof(trainedIterations));

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

void loadTheta(vector<MatrixXd>& theta)
{
	std::ifstream f(thetaFile.c_str(), std::ios::binary);

	size_t thetas;
	f.read((char *)&thetas, sizeof(thetas));

	theta.resize(thetas);

	unsigned int trainedIterationsLocal;
	f.read((char *)&trainedIterationsLocal, sizeof(trainedIterationsLocal));
		
	trainedIterations = trainedIterationsLocal;

	for(unsigned int i = 0; i < thetas; i++)
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

void saveNormalizations(const RowVectorXd& means, RowVectorXd& stdDev)
{
	std::ofstream f(normalizationFile.c_str(), std::ios::binary);

	Eigen::MatrixXd::Index cols;
	cols = means.cols();

	f.write((char *)&cols, sizeof(cols));
	f.write((char *)means.data(), sizeof(Eigen::RowVectorXd::Scalar) * cols);

	cols = stdDev.cols();
	f.write((char *)&cols, sizeof(cols));
	f.write((char *)stdDev.data(), sizeof(Eigen::RowVectorXd::Scalar) * cols);

	f.close();
}

void loadNormalizations(RowVectorXd& means, RowVectorXd& stdDev)
{
	std::ifstream f(normalizationFile.c_str(), std::ios::binary);

	Eigen::MatrixXd::Index cols;
	
	f.read((char *)&cols, sizeof(cols));

	means.resize(cols);

	f.read((char *)means.data(), sizeof(Eigen::MatrixXd::Scalar) * cols);

	if (f.bad())
	{
		throw "Error reading data";
	}

	f.read((char *)&cols, sizeof(cols));

	stdDev.resize(cols);

	f.read((char *)stdDev.data(), sizeof(Eigen::MatrixXd::Scalar) * cols);

	if (f.bad())
	{
		throw "Error reading matrix";
	}
	
	f.close();
}

/*void savePca(const PrincipalComponentAnalysis& pca)
{
	std::ofstream f(pcaFile.c_str(), std::ios::binary);

	Eigen::MatrixXd::Index rows, cols;
	rows = pca.getMatrixUReduce().rows();
	cols = pca.getMatrixUReduce().cols();

	f.write((char *)&rows, sizeof(rows));
	f.write((char *)&cols, sizeof(cols));
	f.write((char *)pca.getMatrixUReduce().data(), sizeof(Eigen::MatrixXd::Scalar) * rows * cols);

	f.close();
}

void loadPca(PrincipalComponentAnalysis& pca)
{
	std::ifstream f(pcaFile.c_str(), std::ios::binary);

	Eigen::MatrixXd::Index rows, cols;
	
	f.read((char *)&rows, sizeof(rows));
	f.read((char *)&cols, sizeof(cols));

	MatrixXd matrixUReduce(rows, cols);

	matrixUReduce.resize(rows, cols);

	f.read((char *)matrixUReduce.data(), sizeof(Eigen::MatrixXd::Scalar) * rows * cols);

	pca.setMatrixUReduce(matrixUReduce);

	if (f.bad())
	{
		throw "Error reading matrix";
	}

	f.close();
}*/

void parseCsv(const string& file, vector<vector<string> >& input)
{
	input.clear();
	ifstream str(file.c_str());
    string line;
		
	while (getline(str, line))
	{
		vector<string> currentLine;
		std::stringstream lineStream(line);
		std::string cell;

		while(getline(lineStream, cell, ','))
		{
			currentLine.push_back(cell);
		}

		input.push_back(currentLine);		
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

void printIteration(unsigned long, const Eigen::VectorXd& x, double cost, const Eigen::VectorXd& gradient)
{
	trainedIterations++;
	std::cout << "iteration: " << trainedIterations << "   objective: " << cost << std::endl;
}