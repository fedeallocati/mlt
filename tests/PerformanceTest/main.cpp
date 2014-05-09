#include <Eigen/Core>
#include <iostream>
#include <omp.h>

using namespace Eigen;
using namespace std;

const char* maxSimdInstructionSetsInUse()
{
#if defined(EIGEN_VECTORIZE_SSE4_2)
  return "SSE4.2";
#elif defined(EIGEN_VECTORIZE_SSE4_1)
  return "SSE4.1";
#elif defined(EIGEN_VECTORIZE_SSSE3)
  return "SSSE3";
#elif defined(EIGEN_VECTORIZE_SSE3)
  return "SSE3";
#elif defined(EIGEN_VECTORIZE_SSE2)
  return "SSE2";
#else
  return "None";
#endif
}

int main()
{
	int maxThreads = Eigen::nbThreads();
	
	cout << fixed;

	for (int t = maxThreads; t > 0; t--)
	{
		Eigen::setNbThreads(t);	
		cout << Eigen::nbThreads() << ";" << maxSimdInstructionSetsInUse() << ";";

		#ifdef _OPENMP
		cout << "Yes";
		#else
		cout << "No";
		#endif

		cout << ";";

		#ifdef _O2
		cout << "Yes";
		#else
		cout << "No";
		#endif

		cout << ";";

		MatrixXd A = MatrixXd::Random(10000, 1000) * 100;
		MatrixXd B = MatrixXd::Random(1000, 5000) * 100;
	
		double mintime = 9999999999;
		for (int i = 0; i < 25; i++)
		{
			double dtime = omp_get_wtime();
			MatrixXd C = A * B;		
			dtime = omp_get_wtime() - dtime;

			if (dtime < mintime)
			{
				mintime = dtime;
			}
		}

		cout << mintime << endl;
	}

	return 0;
}