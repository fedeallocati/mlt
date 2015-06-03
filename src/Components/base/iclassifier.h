#ifndef ICLASSIFIER_H
#define ICLASSIFIER_H

namespace MLT
{
namespace Base
{	
	using namespace Eigen;

	class IClassifier
	{
	public:
		virtual VectorXi classify(const MatrixXd& features) const = 0;
	};
}
}
#endif // ICLASSIFIER_H