#define EIGEN_USE_MKL_ALL
//#define MLT_VERBOSE_TRAINING

#include <Eigen/Eigen>
#include "datasets.hpp"


extern void lr_example(std::tuple<Eigen::MatrixXd, Eigen::MatrixXd> data, Eigen::VectorXd test);
extern void pca_example(Eigen::MatrixXd data, Eigen::VectorXd test);

int main() {
    lr_example(house_value_dataset(), Eigen::Vector3d(1, 1650, 3));
    lr_example(correlated_data_dataset(1000000), correlatedData(0));
    pca_example(std::get<0>(house_value_dataset()), Eigen::Vector3d(1, 1650, 3));
    pca_example(std::get<0>(correlated_data_dataset(1000000)), correlatedData(0));

    return 0;
}