#ifndef __NEWTON_LINEAR_SOLVE_HPP_
#define __NEWTON_LINEAR_SOLVE_HPP_

#include "system_of_eqs.hpp"

namespace Kadath {
namespace linear_solvers {

/// Linear solver of the Newton-Raphson update.
/// KRYLOV: GMRES on the matrix-free J v, preconditioned by MUMPS factors of an earlier Jacobian (default with
///         MUMPS). MUMPS: exact sparse Newton step every iteration. SCALAPACK: dense LU (default without MUMPS).
enum class NewtonBackend { SCALAPACK, MUMPS, KRYLOV };

/// Backend from KADATH_LINSOLVE ("krylov", "mumps" or "scalapack"; default krylov when MUMPS is compiled in,
/// otherwise scalapack). Read on rank 0 and broadcast so every rank agrees; collective.
NewtonBackend newton_backend_from_env();

/// Newton update dx = J^{-1} second with the dense ScaLAPACK solver; the full dx is returned on every rank.
/// @param second residual from System_of_eqs::sec_member(), after vars_to_terms(); collective.
Array<double> newton_update_scalapack(System_of_eqs& syst,
                                      Array<double> const& second,
                                      System_of_eqs::SOLVER solver);

#ifdef KADATH_USE_MUMPS
/// Newton update dx = J^{-1} second with a sparse Jacobian and MUMPS; the full dx is returned on every rank.
/// Requires as many equations as unknowns; collective.
/// @param status_out if null, a MUMPS failure aborts (like a singular Jacobian on the ScaLAPACK path);
///        otherwise it receives 0 or INFOG(1) and a zero update is returned on failure.
Array<double> newton_update_mumps(System_of_eqs& syst,
                                  Array<double> const& second,
                                  int* status_out = nullptr);

/// Counters of the Newton-Krylov solver since the last reset_newton_krylov().
struct NewtonKrylovStats {
    int krylov_steps =
        0;  ///< Newton steps solved by GMRES with an existing preconditioner.
    int krylov_iterations = 0;  ///< GMRES iterations summed over all steps.
    int refactorizations =
        0;  ///< Exact sparse Newton steps (new preconditioner).
    int gmres_failures =
        0;  ///< GMRES runs that missed the forcing term and caused a refactorization.
};

/// Newton step with GMRES (J v from do_JX) preconditioned by kept MUMPS factors, refactoring when needed;
/// collective.
Array<double> newton_update_krylov(System_of_eqs& syst,
                                   Array<double> const& second);

/// Counters of the Newton-Krylov solver.
NewtonKrylovStats const& newton_krylov_stats();

/// Drops the kept preconditioner and the counters.
void reset_newton_krylov();
#endif

/// A Newton iterate diverges when max|F| exceeds this factor times the largest residual seen for the same system
/// (its initial residual or the one after a parameter change). Newton steps grow the residual by a few times at
/// most when they recover; a diverged iterate (e.g. 1e3 -> 1e23) cannot be repaired by the linear solver.
constexpr double kDivergenceGrowth = 1e6;

/// True if @p residual (max|F|) is not finite or exceeds kDivergenceGrowth times @p largest (the largest finite
/// residual of the system so far; 0 = none yet).
bool newton_diverges(double largest, double residual);

/// Dispatches to the configured backend (GAUSS_NEWTON always uses ScaLAPACK) and prints the peak memory;
/// collective. MUMPS failures are retried along mumps_retry_ladder; if every attempt fails the run aborts. On a
/// diverging iterate (newton_diverges) the sparse solvers try only the configured options and stop if they fail,
/// instead of the fallbacks, which need more memory (up to out-of-core) and cannot repair the iteration; a
/// non-finite residual stops the run.
Array<double> newton_update(System_of_eqs& syst,
                            Array<double> const& second,
                            System_of_eqs::SOLVER solver);

}  // namespace linear_solvers
}  // namespace Kadath

#endif
