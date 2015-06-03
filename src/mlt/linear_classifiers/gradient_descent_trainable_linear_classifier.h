#ifndef GRADIENT_DESCENT_TRAINABLE_LINEAR_CLASSIFIER_H
#define GRADIENT_DESCENT_TRAINABLE_LINEAR_CLASSIFIER_H

#include <Eigen/Core>

#include "linear_classifier.h"
#include "../base/igradient_descent_trainable.h"

namespace MLT
{
namespace LinearClassifiers
{
	using namespace Eigen;
	using namespace Base;
	
	class GradientDescentTrainableLinearClassifier : public LinearClassifier, public IGradientDescentTrainable
	{
	public:
		inline const VectorXd parameters() const
		{
			return LinearClassifier::parameters();
		}

		inline void set_parameters(const VectorXd& parameters)
		{
			LinearClassifier::set_parameters(parameters);
		}

		inline bool add_intercept() const { return true; }

	protected:
		GradientDescentTrainableLinearClassifier(size_t input, size_t output, double initial_epsilon) :
			LinearClassifier(input, output, initial_epsilon)
		{
		}
	};
}
}
#endif // GRADIENT_DESCENT_TRAINABLE_LINEAR_CLASSIFIER_H