#ifndef MLT_DEFS_HPP
#define MLT_DEFS_HPP

#include <Eigen/Core>

namespace mlt {
	using namespace std;
	using namespace Eigen;

	using VectorXdRef = const Ref<const VectorXd>&;
	using MatrixXdRef = const Ref<const MatrixXd>&;

	using VectorXiRef = const Ref<const VectorXi>&;
	using MatrixXiRef = const Ref<const MatrixXi>&;

	using Features = MatrixXdRef;
}
#endif