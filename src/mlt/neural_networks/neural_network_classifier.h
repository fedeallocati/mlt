#ifndef NEURAL_NETWORK_CLASSIFIER
#define NEURAL_NETWORK_CLASSIFIER

#include <Eigen/Core>

#include "../base/iclassifier.h"
#include "../base/iparameterized.h"

namespace MLT
{
namespace NeuralNetworks
{	
	using namespace Eigen;
	using namespace Base;

	class NeuralNetworkClassifier : public IParameterized, public IClassifier
	{
	public:	
		VectorXi classify(const MatrixXd& features, MatrixXd& confidences) const
		{
			assert(features.cols() == this->_input);

			MatrixXd x(features.rows(), this->_input + 1);
			x.block(0, 1, features.rows(), this->_input) = features;
			x.col(0) = VectorXd::Ones(features.rows());

			confidences = this->_score(x);

			VectorXi classification(confidences.cols());

			for (size_t i = 0; i < confidences.cols(); i++)
			{
				MatrixXd::Index maxRow, maxCol;
				double max = confidences.col(i).maxCoeff(&maxRow, &maxCol);

				classification(i) = (int)maxRow;
			}

			return classification;
		}

		inline VectorXi classify(const MatrixXd& features) const
		{
			MatrixXd confidences;
			return this->classify(features, confidences);
		}

	protected:
		NeuralNetworkClassifier(size_t input, size_t output)
		{
			assert(input > 0);
			assert(output > 1);

			this->_input = input;
			this->_output = output;
		}

		virtual MatrixXd _score(const Eigen::MatrixXd& x) const = 0;
		
		size_t _input;
		size_t _output;
	};
}
}
#endif // NEURAL_NETWORK_CLASSIFIER