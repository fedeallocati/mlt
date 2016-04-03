#ifndef MLT_MODELS_TRANSFORMATIONS_PRINCIPAL_COMPONENTS_ANALYSIS_HPP
#define MLT_MODELS_TRANSFORMANTIONS_PRINCIPAL_COMPONENTS_ANALYSIS_HPP

#include <Eigen/Core>
#include <Eigen/SVD>

#include "../linear_model.hpp"

namespace mlt {
namespace models {
namespace transformations {
	class PrincipalComponentsAnalysis : public BaseModel {
	public:
		explicit PrincipalComponentsAnalysis(int components_size, bool normalize_mean = false, bool normalize_variance = false) : 
			_components_size(components_size), _normalize_mean(normalize_mean), _normalize_variance(normalize_variance) {}
		
		explicit PrincipalComponentsAnalysis(double variance_to_retain, bool normalize_mean = false, bool normalize_variance = false) :
			_variance_to_retain(variance_to_retain), _normalize_mean(normalize_mean), _normalize_variance(normalize_variance) {
			assert(variance_to_retain > 0 && variance_to_retain <= 1);
		}

		int components_size() const { assert(_fitted); return _components_size; }

		Eigen::MatrixXd components() const { assert(_fitted); return _components; }

		Eigen::VectorXd explained_variance_ratio() const { assert(_fitted); return _explained_variance_ratio; }

		PrincipalComponentsAnalysis& fit(const Eigen::MatrixXd& input) {
			auto final = input;

			if (_normalize_mean) {
				_mean = input.colwise().mean();
				final.rowwise() -= _mean.transpose();
			}

			if (_normalize_variance) {
				_std = (input.transpose() * input).diagonal().cwiseSqrt();
				final.array().rowwise() /= _std.transpose().array();
			}

			auto svd = ((input.transpose() * input) / input.rows()).jacobiSvd(Eigen::ComputeThinU);

			_explained_variance_ratio = svd.singularValues() / svd.singularValues().sum();

			if (_components_size < 1 && _variance_to_retain > 0) {
				double acum = 0;

				size_t i = 0;
				while (i < _explained_variance_ratio.rows() && acum < _variance_to_retain) {
					acum += _explained_variance_ratio(i);
					i++;
				}

				_components_size = i + 1;
			} 

			if (_components_size < svd.matrixU().cols()) {
				_components = svd.matrixU().leftCols(_components_size);
				_explained_variance_ratio = _explained_variance_ratio.head(_components_size);
			}
			else {
				_components_size = _components.cols();
				_components = svd.matrixU();
			}

			_fitted = true;
			_input_size = _components.rows();
			_output_size = _components.cols();

			return *this;
		}

		Eigen::MatrixXd transform(const Eigen::MatrixXd& input) const {
			assert(_fitted);

			Eigen::MatrixXd res;

			if (input.cols() == 1 && input.rows() == _input_size) {
				Eigen::VectorXd final = input;
				if (_normalize_mean) {
					final -= _mean;
				}
				if (_normalize_variance) {
					final.array() /= _std.transpose().array();
				}

				res = _components.transpose() * final;
			}
			else {
				Eigen::MatrixXd final = input;
				if (_normalize_mean) {
					final.rowwise() -= _mean.transpose();
				}
				if (_normalize_variance) {
					final.array().rowwise() /= _std.transpose().array();
				}

				res = final * _components;
			}

			return res;
		}

	protected:
		int _components_size = -1;
		double _variance_to_retain = -1;
		bool _normalize_mean = false;
		bool _normalize_variance = false;
		Eigen::VectorXd _mean;
		Eigen::VectorXd _std;
		Eigen::MatrixXd _components;
		Eigen::VectorXd _explained_variance_ratio;
	};
}
}
}
#endif