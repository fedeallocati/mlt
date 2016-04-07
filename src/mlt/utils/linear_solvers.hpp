#ifndef MLT_UTILS_LINEAR_SOLVERS_HPP
#define MLT_UTILS_LINEAR_SOLVERS_HPP

#include <Eigen/Core>
#include <Eigen/IterativeLinearSolvers>
#include <Eigen/Cholesky>
#include <Eigen/SVD>

namespace mlt {
namespace utils {
namespace linear_solvers {

	template <typename Solver, typename SolverImplementation>
	class BaseSolver
	{
	public:
		Solver& compute(const Eigen::MatrixXd& A)
		{
			_solver.compute(A);
			return static_cast<Solver&>(*this);
		}

		Eigen::MatrixXd solve(const Eigen::MatrixXd& B) const
		{
			return _solver.solve(B);
		}
	protected:
		BaseSolver() = default;

		SolverImplementation _solver;
	};

	class CGSolver : public BaseSolver<CGSolver, Eigen::ConjugateGradient<Eigen::MatrixXd>> {};

	class LLTSolver : public BaseSolver<LLTSolver, Eigen::LLT<Eigen::MatrixXd>> {};

	class LDLTSolver : public BaseSolver<LDLTSolver, Eigen::LDLT<Eigen::MatrixXd>> {};

	class SVDSolver : public BaseSolver<SVDSolver, Eigen::JacobiSVD<Eigen::MatrixXd>>
	{
	public:
		SVDSolver& compute(const Eigen::MatrixXd& A)
		{
			_solver.compute(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
			return *this;
		}
	};
}
}
}
#endif