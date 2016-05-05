namespace mlt {
namespace utils	{
	using Features = const Eigen::Ref<const Eigen::MatrixXd>;
	using RegressorTarget = const Eigen::Ref<const Eigen::MatrixXd>;
	using ClassifierTarget = const Eigen::Ref<const Eigen::VectorXi>;
	using RegressorResult = Eigen::MatrixXd;
	using ClassifierResult = Eigen::VectorXi;
}	
}