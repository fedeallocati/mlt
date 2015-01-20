#ifndef SOFTMAX_CLASSIFIER_H
#define SOFTMAX_CLASSIFIER_H

#include "LinearClassifierBase.h"

class SoftmaxClassifier : public LinearClassifierBase
{	
public:
	SoftmaxClassifier(size_t input, size_t output);
	SoftmaxClassifier(const Eigen::MatrixXd& theta);
	
protected:
	double lossInternal(const Eigen::MatrixXd& theta, const Eigen::MatrixXd& x, const Eigen::MatrixXd& y, double lambda);
	Eigen::MatrixXd gradientInternal(const Eigen::MatrixXd& theta, const Eigen::MatrixXd& x, const Eigen::MatrixXd& y, double lambda);
	Eigen::MatrixXd predictInternal(const Eigen::MatrixXd& theta, const Eigen::MatrixXd& x);
	
private:
	Eigen::MatrixXd score(const Eigen::MatrixXd& theta, const Eigen::MatrixXd& x);
};

#endif // SOFTMAX_CLASSIFIER_H