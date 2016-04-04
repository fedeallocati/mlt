#ifndef MLT_MODELS_TRANSFORMATIONS_PRINCIPAL_COMPONENTS_ANALYSIS_HPP
#define MLT_MODELS_TRANSFORMATIONS_PRINCIPAL_COMPONENTS_ANALYSIS_HPP

#include <Eigen/Core>
#include <Eigen/SVD>

#include "../linear_model.hpp"

namespace mlt {
namespace models {
namespace transformations {
	class PrincipalComponentsAnalysis : public BaseModel {
	public:
		explicit PrincipalComponentsAnalysis(int components_size, bool whiten = false) : _components_size(components_size), _whiten(whiten) {}
		
		explicit PrincipalComponentsAnalysis(double variance_to_retain, bool whiten = false) : _variance_to_retain(variance_to_retain), _whiten(whiten) {
			assert(variance_to_retain > 0 && variance_to_retain <= 1);
		}

		explicit PrincipalComponentsAnalysis(bool whiten = false) : _whiten(whiten) {}

		int components_size() const { assert(_fitted); return _components_size; }

		Eigen::MatrixXd components() const { assert(_fitted); return _components; }

		Eigen::VectorXd explained_variance_ratio() const { assert(_fitted); return _explained_variance_ratio; }

		double noise_variance() const { assert(fitted); return _noise_variance; }

		PrincipalComponentsAnalysis& fit(const Eigen::MatrixXd& input) {
			assert(_components_size == -1 || _components_size <= input.cols());
			_mean = input.rowwise().mean();
			Eigen::MatrixXd final =  input.colwise() - _mean;

			auto svd = ((final * final.transpose()) / input.cols()).jacobiSvd(Eigen::ComputeThinU);

			_explained_variance = svd.singularValues();
			_explained_variance_ratio = _explained_variance / _explained_variance.sum();

			if (_components_size < 1 && _variance_to_retain < 0) {
				_components_size = input.rows();
			} else if (_components_size < 1 && _variance_to_retain > 0) {
				double acum = 0;
				size_t i = 0;
				while (i < _explained_variance_ratio.rows() && acum < _variance_to_retain) {
					acum += _explained_variance_ratio(i);
					i++;
				}
				_components_size = i + 1;
			}

			if (_components_size > svd.matrixU().cols()) {
				_components_size = _components.cols();
			}

			if (_components_size < std::min(input.rows(), input.cols())) {
				_noise_variance = _explained_variance.tail(_explained_variance.size() - _components_size).mean();
			}
			else {
				_noise_variance = 0;
			}

			_components = svd.matrixU().leftCols(_components_size);
			_explained_variance = _explained_variance.head(_components_size);
			_explained_variance_ratio = _explained_variance_ratio.head(_components_size);

			_fitted = true;
			_input_size = _components.rows();
			_output_size = _components.cols();

			return *this;
		}

		Eigen::MatrixXd transform(const Eigen::MatrixXd& input) const {
			assert(_fitted);

			if (_whiten) {
				auto transformed = (_components.transpose() * (input.colwise() - _mean));
				return transformed.array().colwise() * _explained_variance.cwiseSqrt().cwiseInverse().array();
			}

			return _components.transpose() * (input.colwise() - _mean);
		}

		Eigen::MatrixXd inverse_transform(const Eigen::MatrixXd& input) const {
			assert(_fitted);

			if (_whiten) {
				return (_components * (input.array().colwise() *_explained_variance.cwiseSqrt().array()).matrix()).colwise() + _mean;
			}

			return (_components * input).colwise() + _mean;
		}

	protected:
		int _components_size = -1;
		double _variance_to_retain = -1;
		Eigen::VectorXd _mean;
		Eigen::MatrixXd _components;
		Eigen::VectorXd _explained_variance;
		Eigen::VectorXd _explained_variance_ratio;
		double _noise_variance = 0;
		bool _whiten;
	};
}
}
}
#endif