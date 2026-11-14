#ifndef __NEWTON_LINEAR_SOLVE_HPP_
#define __NEWTON_LINEAR_SOLVE_HPP_

#include "system_of_eqs.hpp"

namespace Kadath {
namespace linear_solvers {

/// Linear solver used for the Newton-Raphson update.
enum class NewtonBackend { SCALAPACK, MUMPS };

/// Backend from KADATH_LINSOLVE ("mumps" or "scalapack"); MUMPS by default when compiled in.
/// Read on rank 0 and broadcast so every rank agrees; collective.
NewtonBackend newton_backend_from_env();

/// Prints the peak resident memory (VmHWM) per rank, per node and in total on rank 0; collective.
void report_newton_memory();

/// True when KADATH_LINSOLVE_COMPARE=1: solve every step with both backends and print the difference.
bool newton_compare_from_env();

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
#endif

#ifdef KADATH_USE_MUMPS
/// Counters of the Newton-Krylov mode since the last reset_newton_krylov().
struct NewtonKrylovStats {
    int krylov_steps =
        0;  ///< Newton steps solved by GMRES with an existing preconditioner.
    int krylov_iterations = 0;  ///< GMRES iterations summed over those steps.
    int refactorizations =
        0;  ///< Exact sparse Newton steps (new preconditioner).
    int gmres_failures =
        0;  ///< GMRES runs that missed the forcing term and caused a refactorization.
    int trial_successes =
        0;  ///< KADATH_KRYLOV_REFRESH=lazy: refactorizations avoided by a successful GMRES trial.
};

/// Newton step with GMRES (J v from do_JX) preconditioned by kept MUMPS factors, refactoring when needed;
/// selected by KADATH_NEWTON_METHOD=krylov with the MUMPS backend; collective.
Array<double> newton_update_krylov(System_of_eqs& syst,
                                   Array<double> const& second);

/// Counters of the Newton-Krylov mode.
NewtonKrylovStats const& newton_krylov_stats();

/// Drops the kept preconditioner and the counters.
void reset_newton_krylov();
#endif

/// Dispatches to the configured backend; GAUSS_NEWTON always uses ScaLAPACK; collective.
/// MUMPS failures are retried along mumps_retry_ladder; if every attempt fails the run aborts, or the step is
/// redone with dense ScaLAPACK when KADATH_MUMPS_FALLBACK=scalapack.
Array<double> newton_update(System_of_eqs& syst,
                            Array<double> const& second,
                            System_of_eqs::SOLVER solver);

}  // namespace linear_solvers
}  // namespace Kadath

#endif
