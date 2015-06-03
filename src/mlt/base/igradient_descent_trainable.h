#ifndef IGRADIENT_DESCENT_TRAINABLE_H
#define IGRADIENT_DESCENT_TRAINABLE_H

#include <Eigen/Core>

#include "iparameterized.h"

namespace MLT
{
namespace Base
{
	using namespace Eigen;

	class IGradientDescentTrainable : public IParameterized
	{
	public:
		virtual double loss(const VectorXd& parameters, const MatrixXd& x, const MatrixXd& y) const = 0;

		virtual VectorXd gradient(const VectorXd& parameters, const MatrixXd& x, const MatrixXd& y) const = 0;

		virtual bool add_intercept() const = 0;
	};
}
}
#endif // IGRADIENT_DESCENT_TRAINABLE_H