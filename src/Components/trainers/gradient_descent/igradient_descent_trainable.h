#ifndef IGRADIENT_DESCENT_TRAINABLE_H
#define IGRADIENT_DESCENT_TRAINABLE_H

#include "../../base/imlt_base.h"
#include "../../base/iparameterized.h"

namespace MLT
{
namespace Trainers
{
namespace GradientDescent
{	
	using namespace Eigen;
	using namespace Base;

	class IGradientDescentTrainable : public IParameterized, public IMLTBase
	{	
	public:
		virtual double loss(const VectorXd& parameters, const MatrixXd& x, const MatrixXd& y) const = 0;

		virtual VectorXd gradient(const VectorXd& parameters, const MatrixXd& x, const MatrixXd& y) const = 0;
	};
}
}
}
#endif // IGRADIENT_DESCENT_TRAINABLE_H