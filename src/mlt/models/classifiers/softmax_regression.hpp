#ifndef LEAST_SQUARES_LINEAR_REGRESSOR_HPP
#define LEAST_SQUARES_LINEAR_REGRESSOR_HPP

#include <Eigen/Core>
#include <Eigen/SVD>

namespace mlt {
namespace models {
namespace classifiers {
    
    // Implementation of Softmax Regression
    // Categorization: 
    // - Application: Classifier
    // - Parametrization: Parametrized
    // - Method of Training: Derivative-Free, Gradient-Based   
    // - Supervision: Supervised
    class SoftmaxRegression {
    public:         
        SoftmaxRegression() : _init(false) {}

        SoftmaxRegression(size_t input, size_t output) : _init(true), _beta(Eigen::MatrixXd::Zero(input + 1, output)) {}

        // Disable copy constructors
        SoftmaxRegression(const SoftmaxRegression& other) = delete;
        SoftmaxRegression& operator=(const SoftmaxRegression& other) = delete;

        inline size_t input() const {
            assert(_init);
            return _beta.rows() - 1;
        }

        inline size_t output() const {
            assert(_init);
            return _beta.cols();
        }

        inline bool add_intercept() const {
            return true;
        }

        inline bool is_initialized() const {
            return _init;
        }

        inline void init(size_t input, size_t classes) {
            _beta = Eigen::MatrixXd::Zero(input + 1, classes);
            _init = true;
        }

        inline void reset() {
            assert(_init);
            _beta.setZero();
        }

        inline Eigen::VectorXd score_single(const Eigen::VectorXd& input) const {
            assert(_init);
            return _beta.transpose() * input;
        }

        inline Eigen::MatrixXd score_multi(const Eigen::MatrixXd& input) const {
            assert(_init);
            return input * _beta;
        }

        inline size_t params_size() const {
            assert(_init);
            return _beta.size();
        }

        inline Eigen::VectorXd params() const {
            assert(_init);
            return Eigen::Map<const Eigen::VectorXd>(_beta.data(), _beta.size());
        }

        inline void set_params(const Eigen::VectorXd& beta) {
            assert(_init);
            _beta = Eigen::Map<const Eigen::MatrixXd>(beta.data(), _beta.rows(), _beta.cols());         
        }

        inline double cost(const Eigen::MatrixXd& input, const Eigen::MatrixXd& result) const {
            assert(_init);
            return _cost_internal(_beta, input, result);
        }

        inline double cost(const Eigen::VectorXd& beta, const Eigen::MatrixXd& input, const Eigen::MatrixXd& result) const {
            assert(_init);          
            return _cost_internal(Eigen::Map<const Eigen::MatrixXd>(beta.data(), _beta.rows(), _beta.cols()), input, result);
        }
        
        inline std::tuple<double, Eigen::VectorXd> cost_and_gradient(const Eigen::MatrixXd& input, const Eigen::MatrixXd& result) const {
            assert(_init);
            auto c_a_g = _cost_and_gradient_internal(_beta, input, result);
            return std::make_tuple(std::get<0>(c_a_g), Eigen::Map<Eigen::VectorXd>(std::get<1>(c_a_g).data(), std::get<1>(c_a_g).size()));
        }

        inline std::tuple<double, Eigen::VectorXd> cost_and_gradient(const Eigen::VectorXd& beta, const Eigen::MatrixXd& input, const Eigen::MatrixXd& result) const {
            assert(_init);
            auto c_a_g = _cost_and_gradient_internal(Eigen::Map<const Eigen::MatrixXd>(beta.data(), _beta.rows(), _beta.cols()), input, result);
            return std::make_tuple(std::get<0>(c_a_g), Eigen::Map<Eigen::VectorXd>(std::get<1>(c_a_g).data(), std::get<1>(c_a_g).size()));
        }

    protected:
        inline double _cost_internal(const Eigen::MatrixXd& beta, const Eigen::MatrixXd& input, const Eigen::MatrixXd& result) const {                      
            double loss = -_softmax(beta, input).cwiseProduct(result).colwise().sum().array().log().sum() / input.rows();           
            //regularization loss += 0.5 * this->_lambda * (theta.array().pow(2)).sum();
            return loss;
        }

        inline std::tuple<double, Eigen::MatrixXd> _cost_and_gradient_internal(const Eigen::MatrixXd& beta, const Eigen::MatrixXd& input, const Eigen::MatrixXd& result) const {
            Eigen::MatrixXd scores = _softmax(beta, input); // 2000x10
            double loss = -scores.cwiseProduct(result).colwise().sum().array().log().sum() / input.rows();
            //regularization loss += 0.5 * this->_lambda * (theta.array().pow(2)).sum();

            Eigen::MatrixXd d_beta = ((scores.transpose() * input) - (result.transpose() * input)).transpose() / input.rows(); // 10x50
            //regularization d_beta += this->_lambda * theta;

            return std::make_tuple(loss, d_beta);
        }

        inline Eigen::MatrixXd _softmax(const Eigen::MatrixXd& beta, const Eigen::MatrixXd& input) const {
            Eigen::MatrixXd result = input * _beta; // 2000x50 * 50x10
            result.colwise() -= result.rowwise().maxCoeff();
            result = result.array().exp();
            result = result.array().colwise() / result.rowwise().sum().array();

            return result;
        }

        bool _init;
        Eigen::MatrixXd _beta;
    };
}
}
}
#endif