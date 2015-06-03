#ifndef IPARAMETERIZED_H
#define IPARAMETERIZED_H

namespace MLT
{
namespace Base
{	
	using namespace Eigen;

	class IParameterized
	{
	public:
		virtual const VectorXd parameters() const = 0;

		virtual void set_parameters(const VectorXd& parameters) = 0;
	};
}
}
#endif // IPARAMETERIZED_H