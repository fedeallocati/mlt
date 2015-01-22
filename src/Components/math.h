#ifndef MATH_H
#define MATH_H

double sigmoid(double z)
{
	double gz = 1.0 / (1.0 + exp(-z));
	return gz < 1 ? gz : 0.9999999999999999;
}

double sigmoidGradient(double z)
{
	double gz = sigmoid(z);

	return gz * (1 - gz);
}

#endif // MATH_H