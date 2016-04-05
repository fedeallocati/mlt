#ifndef MLT_MODELS_TRANSFORMERS_PRINCIPAL_COMPONENTS_ANALYSIS_IMPL_HPP
#define MLT_MODELS_TRANSFORMERS_PRINCIPAL_COMPONENTS_ANALYSIS_IMPL_HPP

#include "../base_model.hpp"
#include "transformer_mixin.hpp"

namespace mlt {
namespace models {
namespace transformers {
	template <typename Concrete>
	class PrincipalComponentsAnalysisImpl : public BaseModel, public TransformerMixin<Concrete> {
	public:
		int components_size() const { assert(this->_fitted); return this->_components_size; }

		const Eigen::MatrixXd& components() const { assert(this->_fitted); return this->_components; }

		const Eigen::VectorXd& explained_variance_ratio() const { assert(this->_fitted); return this->_explained_variance_ratio; }

		double noise_variance() const { assert(this->_fitted); return this->_noise_variance; }

		Eigen::MatrixXd transform(const Eigen::MatrixXd& input) const {
			assert(this->_fitted);

			if (this->_whiten) {
				auto transformed = (this->_components.transpose() * (input.colwise() - this->_mean));
				return transformed.array().colwise() * this->_explained_variance.cwiseSqrt().cwiseInverse().array();
			}

			return this->_components.transpose() * (input.colwise() - this->_mean);
		}

		Eigen::MatrixXd inverse_transform(const Eigen::MatrixXd& input) const {
			assert(this->_fitted);

			if (this->_whiten) {
				return (this->_components * (input.array().colwise() *this->_explained_variance.cwiseSqrt().array()).matrix()).colwise() + this->_mean;
			}

			return (this->_components * input).colwise() + this->_mean;
		}

		using TransformerMixin<Concrete>::fit;

		Concrete& fit(const Eigen::MatrixXd& input) {
			assert(this->_components_size == -1 || this->_components_size <= input.cols());

			this->_mean = input.rowwise().mean();
			Eigen::MatrixXd final = input.colwise() - this->_mean;

			auto svd = ((final * final.transpose()) / input.cols()).jacobiSvd(Eigen::ComputeThinU);

			this->_explained_variance = svd.singularValues();
			this->_explained_variance_ratio = this->_explained_variance / this->_explained_variance.sum();

			if (this->_components_size < 1 && this->_variance_to_retain < 0) {
				this->_components_size = input.rows();
			}
			else if (this->_components_size < 1 && this->_variance_to_retain > 0) {
				double acum = 0;
				size_t i = 0;
				while (i < this->_explained_variance_ratio.rows() && acum < this->_variance_to_retain) {
					acum += this->_explained_variance_ratio(i);
					i++;
				}
				this->_components_size = i + 1;
			}

			if (this->_components_size > svd.matrixU().cols()) {
				this->_components_size = this->_components.cols();
			}

			if (this->_components_size < std::min(input.rows(), input.cols())) {
				this->_noise_variance = this->_explained_variance.tail(this->_explained_variance.size() - this->_components_size).mean();
			}
			else {
				this->_noise_variance = 0;
			}

			this->_components = svd.matrixU().leftCols(this->_components_size);
			this->_explained_variance = this->_explained_variance.head(this->_components_size);
			this->_explained_variance_ratio = this->_explained_variance_ratio.head(this->_components_size);

			this->_fitted = true;
			this->_input_size = this->_components.rows();
			this->_output_size = this->_components.cols();

			return static_cast<Concrete&>(*this);
		}

	protected:
		explicit PrincipalComponentsAnalysisImpl(int components_size, bool whiten = false) : _components_size(components_size), _whiten(whiten) {}

		explicit PrincipalComponentsAnalysisImpl(double variance_to_retain, bool whiten = false) : _variance_to_retain(variance_to_retain), _whiten(whiten) {
			assert(variance_to_retain > 0 && variance_to_retain <= 1);
		}

		explicit PrincipalComponentsAnalysisImpl(bool whiten = false) : _whiten(whiten) {}

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