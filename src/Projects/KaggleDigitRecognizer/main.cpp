#define EIGEN_VECTORIZE_SSE4_2
#define EIGEN_USE_MKL_ALL

#include <fstream>
#include <iostream>
#include <time.h>
#include <Eigen/Core>
#include <omp.h>

using namespace Eigen;
using namespace std;

#include "../../Components/LinearClassifiers/SoftmaxLinearClassifier.h"
#include "../../Components/NeuralNetworks/MultilayerPerceptron.h"
#include "../../Components/PrincipalComponentAnalysis/CovariancePCA.h"

void NN(vector<vector<int> >& set, bool optimize = false);
void SoftmaxLinear(vector<vector<int> >& set, bool optimize);

void saveTheta(vector<MatrixXd>& theta, const char* file);
void loadTheta(vector<MatrixXd>& theta, const char* file);

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

	vector<vector<int> > set;

	bool mlp = askYesNoQuestion("Use Multilayer Perceptron?");

	cout << "Loading training set" << endl;

	parseCsv("KaggleDigitRecognizer-train.csv", true, set);
	
	if (mlp)
	{
		NN(set, true);
	}
	else
	{
		SoftmaxLinear(set, true);
	}
	
	cin.get();

	return 0;
}

void NN(vector<vector<int> >& set, bool optimize)
{
	size_t totalSize = set.size();	
	size_t features = set[0].size() - 1;
	int hiddenLayer1 = 500;
	int hiddenLayer2 = 500;
	size_t iterations = 250;
	double lambda = 1;

	MatrixXd trainingSet(totalSize, features);
	VectorXi trainingLabels(totalSize);

	cout << "Moving training set to Eigen" << endl << endl;

	for (unsigned int i = 0; i < totalSize; i++)
	{
		trainingLabels(i) = set[i][0];

		for (unsigned int j = 1; j < features + 1; j++)
		{
			trainingSet(i, j - 1) = set[i][j] - 128;
		}

		set[i].clear();
		vector<int>().swap(set[i]);
		set[i].clear();
	}

	set.clear();
	vector<vector<int>>().swap(set);
	set.clear();

	double maxVal = trainingSet.maxCoeff();

	trainingSet = trainingSet / maxVal;

	cout << "Training PCA" << endl;

	CovariancePCA pca;
	
	double dtime = omp_get_wtime();

	pca.train(trainingSet);

	dtime = omp_get_wtime() - dtime;

	cout << "Training time: " << dtime << "s" << endl << endl;

	if(optimize)
	{	
		size_t trainingSize = totalSize * 3 / 4;
		size_t crossValSize = totalSize - trainingSize;
				
		MatrixXd projectedTrainingSet = pca.transform(trainingSet.topRows(trainingSize));
		MatrixXd projectedCrossValSet = pca.transform(trainingSet.bottomRows(crossValSize));

		double lambdas[] = { 0.0001, 0.001, 0.01, 1, 10 };
		int hiddenLayers1[] = { 250, 500, 1000 };
		int hiddenLayers2[] = { 0, 250, 500, 1000 };
		int lambdasSize = sizeof(lambdas) / sizeof(double);
		int hiddenLayers1Size = sizeof(hiddenLayers1) / sizeof(int);
		int hiddenLayers2Size = sizeof(hiddenLayers2) / sizeof(int);

		double maxHit = -1;
		hiddenLayer1 = -1;
		hiddenLayer2 = -1;
		lambda = -1;

		string file = "mlp-optimization-" + currentDateTime() + ".out";

		cout << "Choosing best hyperparameters with Cross-Validation" << endl;

		for(int i = 0; i < hiddenLayers1Size; i++)
		{
			for (int j = 0; j < hiddenLayers2Size; j++)
			{
				for (int k = 0; k < lambdasSize; k++)
				{
					vector<size_t> layers;

					cout << "Training architecture: [" << pca.getNumberOfFeatures() << " - "
						<< hiddenLayers1[i] << " - " << hiddenLayers2[j] << " - " 
						<< 10 << "] and lambda: " << lambdas[k] << endl;

					if (hiddenLayers1[i] != 0)
					{
						layers.push_back(hiddenLayers1[i]);
					}					

					if (hiddenLayers2[j] != 0)
					{
						layers.push_back(hiddenLayers2[j]);
					}
					
					MultilayerPerceptron nn(pca.getNumberOfFeatures(), layers, 10, 0.12, lambdas[k]);

					LBFGS searchStrategy(50);
					ObjectiveDelta stopStrategy(1e-7, iterations);

					nn.train(projectedTrainingSet, trainingLabels, searchStrategy, stopStrategy);
										
					size_t trainHit = nn.predict(projectedTrainingSet)
						.cwiseEqual(trainingLabels.topRows(trainingSize)).count();
					size_t crossValHit = nn.predict(projectedCrossValSet)
						.cwiseEqual(trainingLabels.bottomRows(crossValSize)).count();

					cout << "Traning set accuracy: " << trainHit / (double)trainingSize << ". Cross-Validation set accuracy: "
						<< crossValHit / (double)crossValSize << endl;

					ofstream optimizeFile(file.c_str(), std::ofstream::app);
					
					optimizeFile << hiddenLayer1 << "-" << hiddenLayer2 << ";" << lambdas[k] << ";" << 
						trainHit / (double)trainingSize << crossValHit / (double)crossValSize << endl;

					optimizeFile.close();

					if (crossValHit > maxHit)
					{
						maxHit = crossValHit;
						hiddenLayer1 = hiddenLayers1[i];
						hiddenLayer2 = hiddenLayers2[i];
						lambda = lambdas[k];
					}
				}
			}
		}
		
		cout << "Best architecture: [" << pca.getNumberOfFeatures() << " - "
			<< hiddenLayer1 << " - " << hiddenLayer2 << " - "
			<< 10 << "] and lambda: " << lambda
			<< ", with accuracy: " << maxHit / (double)crossValSize << endl << endl;
	}

	cout << "Training for evaluation with architecture: [" << pca.getNumberOfFeatures() << " - "
		<< hiddenLayer1 << " - " << hiddenLayer2 << " - "
		<< 10 << "] and lambda: " << lambda << endl;

	vector<size_t> layers;

	if (hiddenLayer1 != 0)
	{
		layers.push_back(hiddenLayer1);
	}

	if (hiddenLayer2 != 0)
	{
		layers.push_back(hiddenLayer2);
	}
		
	MultilayerPerceptron nn(pca.getNumberOfFeatures(), layers, 10, 0.12, lambda);
	LBFGS searchStrategy(50);
	ObjectiveDelta stopStrategy(1e-7, iterations);

	dtime = omp_get_wtime();

	nn.train(pca.transform(trainingSet), trainingLabels, searchStrategy, stopStrategy);

	dtime = omp_get_wtime() - dtime;

	cout << "Training time: " << dtime << "s" << endl << endl;
	
	cin.get();

	cout << "Loading test set" << endl;
	
	parseCsv("KaggleDigitRecognizer-test.csv", true, set);

	MatrixXd testSet(set.size(), features);

	cout << "Moving test set to Eigen" << endl << endl;

	for (size_t i = 0; i < set.size(); i++)
	{
		for (size_t j = 0; j < set[i].size(); j++)
		{
			testSet(i, j) = set[i][j] - 128;
		}

		set[i].clear();
		vector<int>().swap(set[i]);
		set[i].clear();
	}

	set.clear();
	vector<vector<int>>().swap(set);
	set.clear();

	testSet = testSet / maxVal;

	cout << "Doing predictions" << endl;

	//FEDE FIX
	VectorXi predictions = static_cast<NeuralNetworkBase<MultilayerPerceptron>>(nn).predict(pca.transform(testSet));

	stringstream ss;

	ss << "mlp-output" << "-";

	for (size_t l = 0; l < layers.size() - 1; l++)
	{
		ss << layers[l] << ".";
	}
	
	ss << layers[layers.size() - 1] << "-" << lambda << "-" << iterations << ".out";

	cout << "Outputing to file " << ss.str() << endl;

	ofstream outputFile(ss.str().c_str(), std::ofstream::app);
	
	outputFile << "ImageId,Label" << endl;

	for (size_t i = 0; i < predictions.rows(); i++)
	{
		outputFile << i + 1 << "," << predictions(i) << endl;
	}

	outputFile.close();

	cout << "Finished" << endl;
}

void SoftmaxLinear(vector<vector<int> >& set, bool optimize)
{
	size_t totalSize = set.size();
	size_t features = set[0].size() - 1;	
	size_t iterations = 500;
	double lambda = 1;

	MatrixXd trainingSet(totalSize, features);
	VectorXi trainingLabels(totalSize);

	cout << "Moving training set to Eigen" << endl << endl;

	for (unsigned int i = 0; i < totalSize; i++)
	{
		trainingLabels(i) = set[i][0];

		for (unsigned int j = 1; j < features + 1; j++)
		{
			trainingSet(i, j - 1) = set[i][j] - 128;
		}

		set[i].clear();
		vector<int>().swap(set[i]);
		set[i].clear();
	}

	set.clear();
	vector<vector<int>>().swap(set);
	set.clear();

	double maxVal = trainingSet.maxCoeff();

	trainingSet = trainingSet / maxVal;

	cout << "Training PCA" << endl;

	CovariancePCA pca;

	double dtime = omp_get_wtime();

	pca.train(trainingSet);

	dtime = omp_get_wtime() - dtime;

	cout << "Training time: " << dtime << "s" << endl << endl;

	if (optimize)
	{
		size_t trainingSize = totalSize * 3 / 4;
		size_t crossValSize = totalSize - trainingSize;

		MatrixXd projectedTrainingSet = pca.transform(trainingSet.topRows(trainingSize));
		MatrixXd projectedCrossValSet = pca.transform(trainingSet.bottomRows(crossValSize));

		double lambdas[] = { 3e-5, 5e-5, 8e-5, 1e-4, 3e-4, 5e-4, 8e-4, 1e-3 ,3e-3 };
		int lambdasSize = sizeof(lambdas) / sizeof(double);	

		double maxHit = -1;		
		lambda = -1;

		string file = "softmax-optimization-" + currentDateTime() + ".out";

		cout << "Choosing best hyperparameters with Cross-Validation" << endl;
		
		for (int i = 0; i < lambdasSize; i++)
		{
			cout << "Training with lambda " << lambdas[i] << endl;

			SoftmaxLinearClassifier cl(pca.getNumberOfFeatures(), 10, 0.001, lambdas[i]);
			
			BFGS searchStrategy;
			ObjectiveDelta stopStrategy(1e-7, iterations);

			cl.train(projectedTrainingSet, trainingLabels, searchStrategy, stopStrategy);

			//FEDE FIX
			size_t trainHit = static_cast<LinearClassifierBase<SoftmaxLinearClassifier>>(cl).predict(projectedTrainingSet)
				.cwiseEqual(trainingLabels.topRows(trainingSize)).count();
			size_t crossValHit = static_cast<LinearClassifierBase<SoftmaxLinearClassifier>>(cl).predict(projectedCrossValSet)
				.cwiseEqual(trainingLabels.bottomRows(crossValSize)).count();

			cout << "Traning set accuracy: " << trainHit / (double)trainingSize << ". Cross-Validation set accuracy: "
				<< crossValHit / (double)crossValSize << endl;

			ofstream optimizeFile(file.c_str(), std::ofstream::app);
						
			optimizeFile << lambdas[i] << ";" << trainHit / (double)trainingSize << crossValHit / (double)crossValSize << endl;

			optimizeFile.close();

			if (crossValHit > maxHit)
			{
				maxHit = crossValHit;
				lambda = lambdas[i];
			}
		}
			
		cout << "Best lambda: " << lambda << ", with accuracy: " << maxHit / (double)crossValSize << endl << endl;
	}

	cout << "Training for evaluation with lambda " << lambda << endl;

	SoftmaxLinearClassifier cl(pca.getNumberOfFeatures(), 10, 0.001, lambda);
	LBFGS searchStrategy(50);
	ObjectiveDelta stopStrategy(1e-7, iterations);

	dtime = omp_get_wtime();

	cl.train(pca.transform(trainingSet), trainingLabels, searchStrategy, stopStrategy);

	dtime = omp_get_wtime() - dtime;

	cout << "Training time: " << dtime << "s" << endl << endl;

	cin.get();

	cout << "Loading test set" << endl;

	parseCsv("KaggleDigitRecognizer-test.csv", true, set);

	MatrixXd testSet(set.size(), features);

	cout << "Moving test set to Eigen" << endl << endl;

	for (size_t i = 0; i < set.size(); i++)
	{
		for (size_t j = 0; j < set[i].size(); j++)
		{
			testSet(i, j) = set[i][j] - 128;
		}

		set[i].clear();
		vector<int>().swap(set[i]);
		set[i].clear();
	}

	set.clear();
	vector<vector<int>>().swap(set);
	set.clear();

	testSet = testSet / maxVal;

	cout << "Doing predictions" << endl;

	//FEDE FIX
	VectorXi predictions = static_cast<LinearClassifierBase<SoftmaxLinearClassifier>>(cl).predict(pca.transform(testSet));

	stringstream ss;

	ss << "softmax-output" << "-";	

	ss << lambda << "-" << iterations << ".out";

	cout << "Outputing to file " << ss.str() << endl;

	ofstream outputFile(ss.str().c_str(), std::ofstream::app);

	outputFile << "ImageId,Label" << endl;

	for (size_t i = 0; i < predictions.rows(); i++)
	{
		outputFile << i + 1 << "," << predictions(i) << endl;
	}

	outputFile.close();

	cout << "Finished" << endl;
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