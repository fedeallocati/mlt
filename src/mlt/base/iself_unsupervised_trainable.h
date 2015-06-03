#ifndef ISELF_UNSUPERVISED_TRAINABLE_H
#define ISELF_UNSUPERVISED_TRAINABLE_H

namespace MLT
{
namespace Base
{	
	using namespace Eigen;

	class ISelfUnsupervisedTrainable
	{
	public:
		virtual void train(const MatrixXd& trainingSet) = 0;
	};
}
}
#endif // ISELF_UNSUPERVISED_TRAINABLE_H