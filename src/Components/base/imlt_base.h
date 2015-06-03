#ifndef IMLT_BASE_H
#define IMLT_BASE_H

namespace MLT
{
namespace Base
{	
	using namespace Eigen;

	class IMLTBase
	{
	public:
		virtual bool add_intercept() const = 0;
	};
}
}
#endif // IMLT_BASE_H