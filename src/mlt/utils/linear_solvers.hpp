#ifndef MLT_UTILS_LINEAR_SOLVERS_HPP
#define MLT_UTILS_LINEAR_SOLVERS_HPP

#include <Eigen/Core>
#include <Eigen/Cholesky>
#include <Eigen/IterativeLinearSolvers>
#include <Eigen/SVD>

#include "../defs.hpp"

namespace mlt {
namespace utils {
namespace linear_solvers {
	template <typename ConcreteSolver, typename SolverImplementation>
	class BaseSolver {
	public:
		ConcreteSolver& compute(MatrixXdRef A) {
			_solver.compute(A);
			return static_cast<ConcreteSolver&>(*this);
		}

		auto solve(MatrixXdRef B) const {
			return MatrixXd{_solver.solve(B)};
		}

	protected:
		BaseSolver() = default;

		SolverImplementation _solver;
	};

	class CGSolver : public BaseSolver<CGSolver, ConjugateGradient<MatrixXd>> {};

	class LLTSolver : public BaseSolver<LLTSolver, LLT<MatrixXd>> {};

	class LDLTSolver : public BaseSolver<LDLTSolver, LDLT<MatrixXd>> {};

	class SVDSolver : public BaseSolver<SVDSolver, JacobiSVD<MatrixXd>>	{
	public:
		SVDSolver& compute(MatrixXdRef A) {
			_solver.compute(A, ComputeThinU | ComputeThinV);
			return *this;
		}
	};
}
}
}
#endif