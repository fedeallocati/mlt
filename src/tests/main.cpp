#define EIGEN_USE_MKL_ALL
#define BOOST_TEST_MODULE test_one

#include <Eigen/Core>

#include <boost/test/included/unit_test.hpp>

using namespace std;

BOOST_AUTO_TEST_CASE(test_one) {
	auto input = Eigen::MatrixXd{ 10, 20 };
	
	BOOST_CHECK(input.rows() == 10);
	BOOST_CHECK(input.cols() == 20);
}