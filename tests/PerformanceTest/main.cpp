#include <Eigen/Core>
#include <iostream>
#include "tiempo.h"

using namespace Eigen;
using namespace std;

int main()
{
	cout << "#Threads: " << Eigen::nbThreads() << endl;
	cout << "SIMD Instruction Sets In Use: " << Eigen::SimdInstructionSetsInUse() << endl;

	MatrixXd A = MatrixXd::Random(6000, 800) * 100;
	MatrixXd B = MatrixXd::Random(800, 5000) * 100;
	
	//unsigned long long int start, end;

	//MEDIR_TIEMPO_START(start);
	cout << "Starting Matrix Product" << endl;
	MatrixXd C = A * B;
	cout << "Ended Matrix Product" << endl;
	//MEDIR_TIEMPO_STOP(end);

	//unsigned long long int ciclos = end - start;
	//cout << fixed << ciclos << endl;

	return 0;
}