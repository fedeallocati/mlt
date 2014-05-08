#include "NeuralNetworkFunctions.h"

double sigmoid(double z)
{
	return 1.0 / (1.0 + exp(-z));
}

double sigmoidGradient(double z)
{
	double gz = sigmoid(z);

	return gz * (1 - gz);
}