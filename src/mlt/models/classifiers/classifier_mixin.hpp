#ifndef MLT_MODELS_CLASSIFIERS_CLASSIFIER_MIXIN_HPP
#define MLT_MODELS_CLASSIFIERS_CLASSIFIER_MIXIN_HPP

#include <Eigen/Core>
#include <vector>

namespace mlt {
namespace models {
namespace classifiers {
	template <class Classifier>
	class ClassifierMixin {
	public:
		/*double score(X, y, sample_weight = None) {
			"""Returns the mean accuracy on the given test data and labels.
				In multi - label classification, this is the subset accuracy
				which is a harsh metric since you require for each sample that
				each label set be correctly predicted.
				Parameters
				----------
				X : array - like, shape = (n_samples, n_features)
				Test samples.
				y : array - like, shape = (n_samples) or (n_samples, n_outputs)
				True labels for X.
				sample_weight : array - like, shape = [n_samples], optional
				Sample weights.
				Returns
				------ -
				score : float
				Mean accuracy of self.predict(X) wrt.y.
				"""
				from.metrics import accuracy_score
				return accuracy_score(y, self.predict(X), sample_weight = sample_weight)
		}*/
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

		/*Classifier& fit(const Eigen::Ref<const Eigen::MatrixXd> input, const Eigen::Ref<const Eigen::MatrixXi>& classes, bool cold_start = true) {
			assert(input.cols() == classes.cols());
			assert((classes.rowwise().sum().array() == 1).all());

			std::vector<uint32_t> classes_vector(classes.cols());

			for (size_t col = 0; col < classes.cols(); col++) {
				for (size_t row = 0; row < classes.rows(); row++) {
					if (classes(row, col) == 1) {
						classes_vector[col] = (row);
						break;
					}
				}
			}

			return this->get_concrete().fit(input, classes_vector, cold_start);
		}*/

	private:
		Classifier& _concrete() const { return static_cast<Classifier&>(*this); }
	};
}
}
}
#endif