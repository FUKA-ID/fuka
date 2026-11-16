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
/// MumpsOptions::out_of_core values.
constexpr int kOutOfCoreNever = 0;
constexpr int kOutOfCoreAuto = 1;
constexpr int kOutOfCoreAlways = 2;

struct MumpsOptions {
    /// Gather the matrix on rank 0 (ICNTL(18)=0) so MUMPS can compute the zero-free-diagonal permutation
    /// ICNTL(6) itself; otherwise use distributed assembled input (ICNTL(18)=3), which avoids the copy on
    /// rank 0 but has no permutation unless matching_entries > 0.
    bool centralized = true;
    /// ICNTL(28)=2 with ParMETIS (ICNTL(29)=2) instead of the sequential METIS ordering (ICNTL(7)=5). Off by
    /// default: ParMETIS 4.0.3 with METIS 5.1.0 aborts with SIGFPE (a division by zero in METIS
    /// FM_2WayCutRefine) on some FUKA Jacobians, also at n = 6k-52k on 80 ranks; the signal cannot be caught.
    bool parallel_analysis = false;
    /// ICNTL(7) of the sequential analysis: 5 = METIS, 2 = AMF (internal). METIS and ParMETIS here use 32-bit
    /// indices and refuse graphs with more than 2^31 - 1 entries (INFOG(1) = -51, e.g. KS binaries with finer
    /// hole shells than (11, 11, 20)); with_internal_ordering() switches to AMF, which has no such limit but
    /// gives more fill-in.
    int ordering = 5;
    /// It is also skipped when the graph of A + A^T may exceed 2^31 entries (2 nnz > 2^31 - 1): MUMPS's
    /// parallel analysis fails there even with 64-bit orderings, the sequential METIS ordering does not.
    /// parallel_analysis applies only from this n on; smaller matrices use the sequential ordering. On small
    /// graphs ParMETIS can hand METIS an empty local subgraph, and METIS 5.1 then divides by zero
    /// (FM_2WayCutRefine, SIGFPE): seen for n = 243 on 4 ranks. The sequential analysis is cheap there anyway.
    int parallel_analysis_min_n = 20000;
    /// ICNTL(29) of the parallel analysis: 1 = PT-Scotch (if MUMPS was built with it), 2 = ParMETIS. PT-Scotch by
    /// default: ParMETIS 4.0.3 aborts with SIGFPE on FUKA Jacobians also with 64-bit indices; PT-Scotch 7.0.10
    /// ran BBH/BHNS cases at 80 ranks without failures.
    int parallel_ordering = 1;
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
    /// Distributed input only: compute the column permutation of ICNTL(6) here (MUMPS applies it only to a
    /// matrix centralized on rank 0). Each rank keeps, of each of its columns, the matching_entries largest
    /// entries and the matching_entries largest relative to their row maxima; rank 0 gathers only those and
    /// computes a maximum-product matching. If they hold no perfect matching, rank 0 also gathers all entries
    /// of the columns its failed searches reached, for up to 8 rounds and 25 % of the matrix; then
    /// kNoMatching is returned and the retry ladder gathers the matrix. 0 = no permutation: Kadath Jacobians
    /// then need 7-25x larger factors and many delayed pivots.
    /// Off by default: on FUKA binaries (2026-11-15, 80 ranks) the sampled entries never held a perfect
    /// matching and the expansion gathered 64-91 % of the matrix on rank 0, so it saved no memory, and the
    /// matching took longer than MUMPS's own analysis (runs 1.2-2.2x slower than centralized input).
    int matching_entries = 0;
    /// Distributed input with matching_entries > 0: also gather, of every row, the matching_row_entries entries
    /// with the largest |a_ij| / max_k |a_kj| over all ranks (one MPI reduction of n x k candidates), so every
    /// equation has candidates in the sample. 0 = column sample only.
    int matching_row_entries = 0;
    /// Distributed input with matching_entries > 0: when the sample holds no perfect matching, the columns reached
    /// by the failed searches send their matching_growth next-largest entries (doubled every round, up to 16
    /// rounds) instead of all their entries (0).
    int matching_growth = 0;
    /// Reuse the matching and potentials of the previous sampling round (only the unmatched columns are searched).
    bool matching_warm_start = true;
    /// Diagnostics only: domain of every row / column (e.g. System_of_eqs::condition_domains / unknown_domains); a
    /// failed sampling round then reports its unmatched rows and columns per domain. Empty = totals only.
    std::vector<int> row_domains, col_domains;
    /// CNTL(7) of the block low-rank factorization (ICNTL(35) = 2): off-diagonal blocks of the fronts are
    /// compressed to this relative accuracy, which cuts factorization flops and factor memory on large 3-D-like
    /// fronts. 0 = full-rank factors. The factors then belong to a perturbed matrix, so only preconditioners
    /// should use it, with residual_tolerance well above blr_tolerance.
    double blr_tolerance = 0.;
    /// Out-of-core factorization (ICNTL(22) = 1): the factors are written to files in ooc_dir and every later
    /// solve (also JOB = 3, each GMRES preconditioner application) reads them back, so it is slower. It lowers
    /// the factorization memory of every rank to the active fronts, not the analysis memory of rank 0.
    ///   kOutOfCoreNever: if the in-core estimate does not fit, solve() returns kInsufficientMemory;
    ///   kOutOfCoreAuto: out of core only when the in-core estimate (largest per rank) exceeds 80 % of the free
    ///     memory per rank, or when the factorization still runs out of memory at that limit; logged on rank 0;
    ///   kOutOfCoreAlways: always out of core.
    int out_of_core = kOutOfCoreAuto;
    /// Directory of the out-of-core files; empty = $TMPDIR, else /tmp. Node-local disks are fastest.
    std::string ooc_dir;
    /// Free memory per rank (MB) for the out-of-core decision and the ICNTL(23) cap; 0 = measured
    /// (MemAvailable / ranks on the node, minimum over the nodes). Set by tests.
    long available_mem_mb = 0;
};

/// Statistics of the last MumpsSolver::solve call.
struct MumpsStats {
    std::int64_t nnz = 0;  ///< Global number of entries passed to MUMPS.
    std::int64_t factor_entries = 0;  ///< INFOG(29): entries in the LU factors.
    std::int64_t compressed_factor_entries =
        0;  ///< INFOG(35): entries stored with BLR compression (= factor_entries without BLR).
    double factor_flops = 0.;  ///< RINFOG(3): flops of the elimination.
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
    bool out_of_core =
        false;  ///< The factors of the last solve were written to disk (ICNTL(22) = 1).
    long mem_budget_mb =
        0;  ///< ICNTL(23) of the last factorization attempt (MB per rank).
    double time_gather =
        0.;  ///< Seconds spent gathering triplets on rank 0 (centralized input).
    double time_matching =
        0.;  ///< Seconds for the column permutation of distributed input (pruning, gather, matching).
    double time_analysis = 0.;  ///< Seconds in JOB=1.
    double time_factor_solve =
        0.;  ///< Seconds in the successful JOB=5 (and failed attempts).
};

/// Status returned by MumpsSolver::solve when MUMPS succeeded but the solution fails the residual check.
constexpr int kInaccurateSolution = -1000;
/// Status returned by MumpsSolver::solve when no perfect matching was found for the column permutation.
constexpr int kNoMatching = -1001;
/// Status returned by MumpsSolver::solve when the factors do not fit in memory and out_of_core = kOutOfCoreNever.
constexpr int kInsufficientMemory = -1002;

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
    std::vector<int>
        col_perm_;  ///< original column (1-based) of permuted column i; empty = no permutation
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

/// INFOG(1) of an analysis whose graph is too large for the 32-bit external ordering.
constexpr int kGraphTooLarge = -51;
/// @p options with the internal AMF ordering and sequential analysis (the fallback after kGraphTooLarge).
MumpsOptions with_internal_ordering(MumpsOptions options);

}  // namespace linear_solvers
}  // namespace Kadath

#endif  // KADATH_USE_MUMPS
#endif
