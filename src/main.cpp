#include "NeuralNetwork.h"
#include "Utils.h"

using namespace Eigen;
using namespace std;

void parseCsv(string file, bool skipFirstLine, vector<vector<int>>& result);

int main()
{
	vector<vector<int>> set;

	parseCsv("E:\\Machine Learning\\Kaggle\\Digit Recognizer\\train.csv", true, set);

	unsigned int features = set[0].size() - 1;

	MatrixXd trainingSet(set.size() * 3 / 4, features);
	VectorXi trainingLabels(set.size() * 3 / 4);

	MatrixXd crossValSet(set.size() * 1 / 4, features);
	VectorXi crossValLabels(set.size() * 1 / 4);

	for(unsigned int i = 0; i < set.size() * 3 / 4 ; i++)
	{
		trainingLabels(i) = set[i][0];

		for(unsigned int j = 1; j < set[i].size(); j++)
		{
			trainingSet(i, j - 1) = set[i][j];
		}
	}

	for(unsigned int i = set.size() * 3 / 4; i < set.size(); i++)
	{
		crossValLabels(i - set.size() * 3 / 4) = set[i][0];

		for(unsigned int j = 1; j < set[i].size(); j++)
		{
			crossValSet(i - set.size() * 3 / 4, j - 1) = set[i][j];
		}
	}

	double maxVal = trainingSet.maxCoeff();

	trainingSet = trainingSet / maxVal;
	crossValSet = crossValSet / maxVal;

	double lambdas[] = {0.03, 0.1, 0.3, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
	int hiddenLayers[] = {5, 25, 50, 100};
	int lambdasSize = sizeof(lambdas) / sizeof(double);
	int hiddenLayersSize = sizeof(hiddenLayers) / sizeof(int);

	double maxAcc = -1;
	double lambda = -1;
	int hiddenLayer = -1;

	for(int i = 0; i < lambdasSize; i++)
	{
		for(int j = 0; j < hiddenLayersSize; j++)
		{
			vector<unsigned int> layers;
			layers.push_back(hiddenLayers[j]);
			NeuralNetwork nn(features, layers, 10);

			nn.train(trainingSet, trainingLabels, lambdas[i]);

			VectorXi predictions = nn.predictMany(crossValSet);
			double acc = (double)predictions.cwiseEqual(crossValLabels).sum() / predictions.rows();

			cout << lambdas[i] << "," << hiddenLayers[j] << "," << acc << endl;

			if(acc > maxAcc)
			{
				maxAcc = acc;
				lambda = lambdas[i];
				hiddenLayer = hiddenLayers[j];
			}
		}
	}
	
	vector<unsigned int> layers;
	layers.push_back(hiddenLayer);
	NeuralNetwork nn(features, layers, 10);

	nn.train(trainingSet, trainingLabels, lambda);

	parseCsv("E:\\Machine Learning\\Kaggle\\Digit Recognizer\\test.csv", true, set);

	MatrixXd testSet(set.size(), features);

	for(unsigned int i = 0; i < set.size(); i++)
	{
		for(unsigned int j = 0; j < set[i].size(); j++)
		{
			testSet(i, j) = set[i][j];
		}
	}

	testSet = testSet / maxVal;

	VectorXi predictions = nn.predictMany(testSet);

	cout << "ImageId,Label" << endl;

	for(unsigned int i = 0; i < predictions.rows(); i++)
	{
		cout << i + 1 << "," << predictions(i) << endl;
	}

	return 0;
}

void parseCsv(string file, bool skipFirstLine, vector<vector<int>>& result)
{
	result.clear();
	ifstream str = ifstream(file);
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