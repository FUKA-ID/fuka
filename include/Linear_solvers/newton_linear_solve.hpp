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

/// Dispatches to the configured backend; GAUSS_NEWTON always uses ScaLAPACK; collective.
/// If MUMPS fails, the step is redone with ScaLAPACK unless KADATH_MUMPS_FALLBACK=abort.
Array<double> newton_update(System_of_eqs& syst,
                            Array<double> const& second,
                            System_of_eqs::SOLVER solver);

}  // namespace linear_solvers
}  // namespace Kadath

#endif
