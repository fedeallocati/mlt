#ifndef MLT_MODELS_CLASSIFIERS_CLASSIFIER_MIXIN_HPP
#define MLT_MODELS_CLASSIFIERS_CLASSIFIER_MIXIN_HPP

#include <Eigen/Core>

namespace mlt {
namespace models {
namespace classifiers {
	template <class Classifier>
	class ClassifierMixin {
	public:
		double score(const Eigen::Ref<const Eigen::MatrixXd>& input, const Eigen::Ref<const Eigen::VectorXi>& classes) const {
			return (this->_concrete().classify(input).array() == classes.array()).cast<double>().sum() / static_cast<double>(input.cols());
		}

	protected:
		ClassifierMixin() {}

		Eigen::MatrixXi _convert_to_classes_matrix(const Eigen::Ref<const Eigen::VectorXi> classes) {
			auto classes_matrix = Eigen::MatrixXi{ Eigen::MatrixXi::Zero(classes.maxCoeff() + 1, classes.size()) };

			for (unsigned int i = 0; i < classes.size(); i++) {
				classes_matrix(classes(i), i) = 1;
			}

			return classes_matrix;
		}

		Eigen::VectorXi _convert_to_classes_vector(const Eigen::Ref<const Eigen::MatrixXi> classes) {
			assert((classes.colwise().sum().array() == 1).all());

			auto classes_vector = Eigen::VectorXi(classes.cols());

			for (size_t col = 0; col < classes.cols(); col++)  {
				for (size_t row = 0; row < classes.rows(); row++) {
					if (classes(row, col) == 1) {
						classes_vector(col) = (row);
						break;
					}
				}
			}

			return classes_vector;
		}

	private:
		const Classifier& _concrete() const { return static_cast<const Classifier&>(*this); }

		Classifier& _concrete() { return static_cast<Classifier&>(*this); }
	};
}
}
}
#endif