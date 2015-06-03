#ifndef LINEAR_CLASSSIFIER_H
#define LINEAR_CLASSSIFIER_H

#include <EigenOptimization/Optimization>
#include "../base/iclassifier.h"
#include "../base/iparameterized.h"

namespace MLT
{
namespace LinearClassifiers
{
	using namespace Eigen;
	using namespace Base;

	class LinearClassifier : public IParameterized, public IClassifier
	{	
	public:
		MatrixXd score(const Eigen::MatrixXd& theta, const Eigen::MatrixXd& x) const
		{	
			return theta * x.transpose();
		}

		VectorXi classify(const MatrixXd& features, MatrixXd& confidences) const
		{
			assert(features.cols() == this->_input);

			MatrixXd x(features.rows(), features.cols() + 1);
			x.block(0, 1, features.rows(), features.cols()) = features;
			x.col(0) = VectorXd::Ones(features.rows());

			confidences = this->score(this->_theta, x);

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

		inline const VectorXd parameters() const
		{			
			return VectorXd::Map(this->_theta.data(), this->_theta.size());
		}

		inline void set_parameters(const VectorXd& parameters)
		{
			assert(parameters.size() == this->_theta.size());
			this->_theta = MatrixXd::Map(parameters.data(), this->_output, this->_input + 1);
		}

		inline bool add_intercept() const { return true; }

		inline const MatrixXd& theta() const
		{
			return this->_theta;
		}

		inline void set_theta(const MatrixXd& theta)
		{
			assert(theta.rows() == this->_output);
			assert(theta.cols() == this->_input + 1);

			this->_theta = theta;
		}

	protected:
		LinearClassifier(size_t input, size_t output, double initial_epsilon)
		{
			assert(input > 0);
			assert(output > 1);

			this->_input = input;
			this->_output = output;

			this->_theta = MatrixXd::Random(output, input + 1) * initial_epsilon;
		}
				
		size_t _input;
		size_t _output;	
		MatrixXd _theta;
	};	
}
}
#endif // LINEAR_CLASSSIFIER_BASE_H