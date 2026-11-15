#ifndef __MUMPS_SOLVER_HPP_
#define __MUMPS_SOLVER_HPP_

#ifdef KADATH_USE_MUMPS

#include <mpi.h>

#include <cstdint>
#include <string>
#include <vector>

#include "Linear_solvers/sparse_triplets.hpp"
#include "dmumps_c.h"

namespace Kadath {
namespace linear_solvers {

/// Settings of MumpsSolver; ICNTL numbers refer to the MUMPS 5.8 user guide. The defaults are the ones the
/// Newton solvers use; the other values exist for the retry ladder and the tests.
struct MumpsOptions {
    /// Gather the matrix on rank 0 (ICNTL(18)=0) so the zero-free-diagonal permutation ICNTL(6) can be used;
    /// otherwise use distributed assembled input (ICNTL(18)=3), which avoids the copy on rank 0.
    bool centralized = true;
    /// ICNTL(28)=2 with ParMETIS (ICNTL(29)=2) instead of the sequential METIS ordering (ICNTL(7)=5).
    bool parallel_analysis = false;
    /// ICNTL(14): initial workspace relaxation, doubled on workspace errors.
    int mem_relax_percent = 30;
    /// Factorization retries after a workspace error (-8, -9, -17, -19, -20).
    int max_mem_retries = 3;
    /// ICNTL(4): 0 silent, 1 errors, 2 + statistics, 3-4 diagnostics.
    int verbosity = 1;
    /// Newton steps retry failed or rejected solves with the other configurations of mumps_retry_ladder.
    bool retry = true;
    /// CNTL(1): relative pivoting threshold. The MUMPS default 0.01 allowed pivot growth to a 1e-11
    /// residual on a Kadath-like test matrix (test_mumps_wrapper); 0.1 restores 1e-16 for ~4 % more fill.
    double pivot_threshold = 0.1;
    /// ICNTL(23): working memory per rank in MB. 0 = automatic: 1.25 x the largest per-rank analysis estimate
    /// INFOG(16) + 64 MB on every rank (a rank with a tiny own estimate that receives delayed pivots then still
    /// has room), at most 80 % of the free node memory per rank; doubled on workspace errors.
    int max_mem_mb = 0;
    /// ICNTL(6): column permutation towards a zero-free diagonal (centralized input only); 7 = automatic, 1 =
    /// structural (the retry). 3 made the analysis 3-4x faster but pivoted badly on large binary Jacobians (up to
    /// 1e5 delayed pivots, residuals 1e-9 ... 4e-7).
    int permutation = 7;
    /// ICNTL(8): scaling strategy; -1 = 0 (none) for centralized input, 77 (automatic) for distributed input.
    /// Centralized + 77 uses the scaling of the ICNTL(6) matching, which produced solutions with residual 1e-2
    /// (reported as success) on BH Jacobians; without scaling they solve to <= 1e-14 at the same fill and time.
    /// Distributed input has no matching and stays accurate with 77 (worse without).
    int scaling = -1;
    /// Largest accepted max|A x - b| / max|b|; a worse solution is rejected with kInaccurateSolution, so a
    /// factorization that reports success but is wrong is never used.
    double residual_tolerance = 1e-10;
};

/// Statistics of the last MumpsSolver::solve call.
struct MumpsStats {
    std::int64_t nnz = 0;  ///< Global number of entries passed to MUMPS.
    std::int64_t factor_entries = 0;  ///< INFOG(29): entries in the LU factors.
    double factor_flops = 0.;         ///< RINFOG(3): flops of the elimination.
    int peak_mem_mb_max =
        0;  ///< INFOG(21): largest per-rank memory actually used (MB).
    int peak_mem_mb_total = 0;  ///< INFOG(22): total memory actually used (MB).
    std::int64_t estimated_factor_entries =
        0;  ///< INFOG(20): analysis estimate of the factor entries.
    int off_diagonal_pivots =
        0;  ///< INFOG(12): pivots chosen off the diagonal (also set on failure).
    int delayed_pivots =
        0;  ///< INFOG(13): pivots delayed by numerical pivoting (also set on failure).
    double relative_residual =
        -1.;  ///< max|A x - b| / max|b| of the returned solution.
    long mem_budget_mb =
        0;  ///< ICNTL(23) of the last factorization attempt (MB per rank).
    double time_gather =
        0.;  ///< Seconds spent gathering triplets on rank 0 (centralized input).
    double time_analysis = 0.;  ///< Seconds in JOB=1.
    double time_factor_solve =
        0.;  ///< Seconds in the successful JOB=5 (and failed attempts).
};

/// Status returned by MumpsSolver::solve when MUMPS succeeded but the solution fails the residual check.
constexpr int kInaccurateSolution = -1000;

/// Unsymmetric sparse direct solve of A x = b with MUMPS on a communicator.
class MumpsSolver {
   public:
    /// Initializes a MUMPS instance (JOB=-1) on @p comm; collective.
    explicit MumpsSolver(MPI_Comm comm,
                         MumpsOptions const& options = MumpsOptions());
    /// Releases the MUMPS instance (JOB=-2); collective.
    ~MumpsSolver();
    MumpsSolver(MumpsSolver const&) = delete;
    MumpsSolver& operator=(MumpsSolver const&) = delete;

    /// Solves A x = b; collective.
    /// @param n global order of A.
    /// @param local this rank's entries of A (1-based); any distribution, duplicates are summed.
    /// @param rhs_solution b on input (read on rank 0, length n on every rank); x on output on every rank.
    /// @return 0 on success, otherwise INFOG(1) (e.g. -10 for a numerically singular matrix) or
    ///         kInaccurateSolution if max|A x - b| / max|b| > residual_tolerance.
    int solve(int n, SparseTriplets& local, std::vector<double>& rhs_solution);

    /// Solves A x = b again with the factors of the last successful solve() (JOB=3), e.g. as a preconditioner
    /// for an updated A; collective. No residual check (the factors may belong to an older matrix).
    /// @param rhs_solution b on input (read on rank 0, length n on every rank); x on output on every rank.
    /// @return 0 on success, -1 without factors, otherwise INFOG(1).
    int solve_factored(std::vector<double>& rhs_solution);

    /// True when the last solve() succeeded, so solve_factored() can be used.
    bool has_factors() const { return factored_; }

    MumpsStats const& stats() const { return stats_; }

   private:
    void run(int job);

    void set_icntl(int i, int value) { id_.icntl[i - 1] = value; }

    int infog(int i) const { return id_.infog[i - 1]; }

    MPI_Comm comm_;
    int rank_ = 0;
    MumpsOptions options_;
    MumpsStats stats_;
    DMUMPS_STRUC_C id_;
    bool factored_ = false;
    int n_ = 0;
};

/// Human-readable message for a MUMPS INFOG(1) error code.
std::string mumps_error_message(int infog1);

/// A named configuration of the MUMPS retry ladder.
struct MumpsAttempt {
    std::string name;
    MumpsOptions options;
};

/// Configurations a Newton step tries in order on the same assembled matrix: @p base, then (if base.retry)
/// the structural permutation ICNTL(6)=1 and distributed input. The retries use the automatic scaling and
/// memory budget, because they are a recovery path; both solved every dumped BH Jacobian accurately.
std::vector<MumpsAttempt> mumps_retry_ladder(MumpsOptions const& base);

}  // namespace linear_solvers
}  // namespace Kadath

#endif  // KADATH_USE_MUMPS
#endif
