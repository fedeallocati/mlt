#ifndef MLT_DEFS_HPP
#define MLT_DEFS_HPP

#include <Eigen/Core>

#ifdef MLT_VERBOSE
#define MLT_LOG(log) cout << log;
#else
#define MLT_LOG(log)
#endif

#define MLT_LOG_LINE(log) MLT_LOG(log << endl);

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