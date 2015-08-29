#ifndef PRINCIPAL_COMPONENTS_ANALYSIS_HPP
#define PRINCIPAL_COMPONENTS_ANALYSIS_HPP

#include <Eigen/Core>
#include <Eigen/SVD>

namespace mlt {
namespace models {
namespace transformations {
    
    // Implementation of Principal Compontents Analysis
    // Categorization: 
    // - Application: Transformation
    // - Parametrization: Non-Parametrized
    // - Method of Training: Self-Trainable
	// - Supervision: Unsupervised
	// Parameters:
	// - bool normalize_mean: indicates wheter to perform mean centering or not
	// - bool normalize_variance: indicates whether to perform variance normalization (divide each feature by it's variance).
	//							  Asumes data has zero mean or normalize_mean has been set to true
	// - int new_dimension: dimension of the projected data. If not set to a positive integer, variance_to_retain is used to
	//						automatically select it. If neither new_dimension nor variance_to_retain are specified will keep, 
	//						full rotation matrix
	// - double variance_to_retain: amount of variance to preserve. This is used for automatic selection of dimension. If
	//								new_dimension was specified, this parameter is ignored. If neither new_dimension nor 
	//								variance_to_retain are specified, will keep full rotation matrix
	template <typename Params>
    class PrincipalComponentsAnalysis {
	public:
		PrincipalComponentsAnalysis() {}

		// Disable copy constructors
		PrincipalComponentsAnalysis(const PrincipalComponentsAnalysis& other) = delete;
		PrincipalComponentsAnalysis& operator=(const PrincipalComponentsAnalysis& other) = delete;

		inline size_t input() const {
			assert(_init);
			return _matrix_u.rows() - 1;
		}

		inline size_t output() const {
			assert(_init);
			return _matrix_u.cols();
		}

		inline bool add_intercept() const {
			return false;
		}

		inline bool is_initialized() const {
			return _init;
		}

		inline void init(size_t input, size_t output) {
			_matrix_u = Eigen::MatrixXd::Zero(input, output);
			_mean = Eigen::VectorXd::Zero(input);
			_std = Eigen::VectorXd::Ones(input);
			_init = true;
		}

		inline void reset() {
			assert(_init);
			_matrix_u.setZero();
			_mean.setZero();
			_std.setZero();
		}

		inline Eigen::VectorXd transform_single(const Eigen::VectorXd& input) const {
			VectorXd final = input;
			if (_normalized_mean) {
				final -= _mean;
			}			
			if (_normalized_variance) {
				final.array() /= _std.transpose().array();
			}
			return _matrix_u.transpose() * final;
        }

		inline Eigen::VectorXd transform_multi(const Eigen::MatrixXd& input) const {
			MatrixXd final = input;
			if (_normalized_mean) {
				final.rowwise() -= _mean.transpose();
			}
			if (_normalized_variance) {
				final.array().rowwise() /= _std.transpose().array();
			}
			return input * _matrix_u;
		}
		
		void self_train(const Eigen::MatrixXd& input, bool reset = false) {
			MatrixXd final = input;
			
			if (params_t::normalize_mean) {
				_mean = input.colwise().mean();
				final.rowwise() -= _mean.transpose();
				_normalized_mean = true;
			} else {
				_mean = Eigen::VectorXd::Zero(input.cols());
			}

			if (params_t::normalize_variance) {
				_std = (input.transpose() * input).diagonal().cwiseSqrt();				
				final.array().rowwise() /= _std.transpose().array();
				_normalized_variance = true;
			} else {
				_std = Eigen::VectorXd::Ones(input.cols());
			}

			Eigen::JacobiSVD<Eigen::MatrixXd> svd = ((input.transpose() * input) / input.rows()).jacobiSvd(Eigen::ComputeThinU);
			auto k = params_t::new_dimension;
			if (k < 1) {
				k = input.cols();
				if (params_t::variance_to_retain > 0 && params_t::variance_to_retain <= 1) {
					double sum = svd.singularValues().sum();
					double acum = 0;

					for (auto i = 0; i < svd.singularValues().rows(); i++) {						
						acum += svd.singularValues()(i);
						if ((acum / sum) >= 0.99) {					
							k = i + 1;
							break;
						}
					}
				}
			}

			_matrix_u = svd.matrixU().leftCols(k);			
			_init = true;
		}	

		void set_mean(const Eigen::VectorXd& mean) {
			assert(_init && _mean.rows() == mean.rows());
			_mean = mean;
			_normalized_mean = true;
		}

		Eigen::VectorXd mean() const {
			assert(_init);
			return _mean;
		}

		void set_std(const Eigen::MatrixXd& std) {			
			assert(_init && _std.rows() == std.rows());
			_std = std;
			_normalized_variance = true;
		}

		Eigen::MatrixXd std() const {
			assert(_init);
			return _std;
		}

		void set_matrix_u(const Eigen::MatrixXd& matrix_u) {
			assert(_init && _matrix_u.rows() == matrix_u.rows() && _matrix_u.cols() == matrix_u.cols());
			_matrix_u = matrix_u;
		}

		Eigen::MatrixXd matrix_u() const {
			assert(_init);
			return _matrix_u;
		}

    protected:
		typedef Params::PrincipalComponentsAnalysis params_t;

		bool _init = false;
		bool _normalized_mean = false;
		Eigen::VectorXd _mean;
		bool _normalized_variance = false;
		Eigen::VectorXd _std;
		Eigen::MatrixXd _matrix_u;
	};
}
}
}
#endif