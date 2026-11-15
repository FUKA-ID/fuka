#ifdef KADATH_USE_MUMPS

#include "Linear_solvers/mumps_solver.hpp"

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace Kadath {
namespace linear_solvers {

namespace {

constexpr int kJobInit = -1;
constexpr int kJobEnd = -2;
constexpr int kJobAnalysis = 1;
constexpr int kJobSolve = 3;
constexpr int kJobFactorSolve = 5;
constexpr int kJobFreeFactors = -4;

/// MUMPS reports large counts as negative numbers of millions.
std::int64_t millions_if_negative(int value) {
    return value >= 0 ? static_cast<std::int64_t>(value)
                      : -static_cast<std::int64_t>(value) * 1000000;
}

bool is_workspace_error(int infog1) {
    return infog1 == -8 || infog1 == -9 || infog1 == -17 || infog1 == -19 ||
           infog1 == -20;
}

/// Smallest per-rank share (MB) of MemAvailable over all nodes of @p comm; 0 if /proc/meminfo is unreadable.
/// Collective. Measured while the caller's data is resident, so it is the memory still free for MUMPS.
long node_memory_share_mb(MPI_Comm comm) {
    MPI_Comm node;
    MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &node);
    int ranks_on_node = 1;
    MPI_Comm_size(node, &ranks_on_node);
    MPI_Comm_free(&node);

    long available_kb = 0;
    std::ifstream meminfo("/proc/meminfo");
    std::string key, unit;
    long value = 0;
    while (meminfo >> key >> value >> unit)
        if (key == "MemAvailable:") {
            available_kb = value;
            break;
        }
    long share = available_kb / 1024 / ranks_on_node;
    long min_share = 0;
    MPI_Allreduce(&share, &min_share, 1, MPI_LONG, MPI_MIN, comm);
    return min_share;
}

int env_int(char const* name, int fallback) {
    char const* value = std::getenv(name);
    return value ? std::atoi(value) : fallback;
}

double env_double(char const* name, double fallback) {
    char const* value = std::getenv(name);
    return value ? std::atof(value) : fallback;
}

std::string env_string(char const* name, std::string const& fallback) {
    char const* value = std::getenv(name);
    return value ? std::string(value) : fallback;
}

template <typename T>
MPI_Datatype mpi_type();

template <>
MPI_Datatype mpi_type<int>() {
    return MPI_INT;
}

template <>
MPI_Datatype mpi_type<double>() {
    return MPI_DOUBLE;
}

/// Concatenates every rank's @p local on rank 0 in rank order; @p counts are the per-rank sizes (valid on rank 0).
/// Messages are chunked so the int count of MPI is never exceeded.
template <typename T>
void gather_on_root(std::vector<T> const& local,
                    std::vector<std::int64_t> const& counts,
                    std::vector<T>& global,
                    MPI_Comm comm) {
    constexpr std::int64_t chunk = 1 << 28;
    int rank, nproc;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nproc);

    if (rank != 0) {
        for (std::int64_t offset = 0;
             offset < static_cast<std::int64_t>(local.size());
             offset += chunk) {
            int const len = static_cast<int>(
                std::min<std::int64_t>(chunk, local.size() - offset));
            MPI_Send(local.data() + offset, len, mpi_type<T>(), 0, 0, comm);
        }
        return;
    }

    std::int64_t total = 0;
    for (auto c : counts)
        total += c;
    global.resize(total);
    std::copy(local.begin(), local.end(), global.begin());

    std::int64_t position = counts[0];
    for (int source = 1; source < nproc; ++source) {
        for (std::int64_t offset = 0; offset < counts[source];
             offset += chunk) {
            int const len = static_cast<int>(
                std::min<std::int64_t>(chunk, counts[source] - offset));
            MPI_Recv(global.data() + position + offset,
                     len,
                     mpi_type<T>(),
                     source,
                     0,
                     comm,
                     MPI_STATUS_IGNORE);
        }
        position += counts[source];
    }
}

}  // namespace

MumpsOptions MumpsOptions::from_env() {
    MumpsOptions options;
    options.centralized = env_string("KADATH_MUMPS_INPUT", "centralized") !=
                          "distributed";

    std::string const ordering = env_string("KADATH_MUMPS_ORDERING", "metis");
    if (ordering == "parmetis")
        options.parallel_analysis = true;
    else if (ordering == "amd")
        options.ordering = 0;
    else if (ordering == "amf")
        options.ordering = 2;
    else if (ordering == "pord")
        options.ordering = 4;
    else if (ordering == "qamd")
        options.ordering = 6;
    else if (ordering == "auto")
        options.ordering = 7;
    else
        options.ordering = 5;

    options.mem_relax_percent =
        env_int("KADATH_MUMPS_MEM_RELAX", options.mem_relax_percent);
    options.verbosity = env_int("KADATH_MUMPS_VERBOSITY", options.verbosity);
    options.omp_threads =
        env_int("KADATH_MUMPS_OMP_THREADS", options.omp_threads);
    options.blr_tolerance =
        env_double("KADATH_MUMPS_BLR", options.blr_tolerance);
    options.out_of_core = env_int("KADATH_MUMPS_OOC", 0) != 0;
    options.retry = env_int("KADATH_MUMPS_RETRY", 1) != 0;
    options.max_mem_mb = env_int("KADATH_MUMPS_MAX_MEM_MB", options.max_mem_mb);
    options.permutation =
        env_int("KADATH_MUMPS_PERMUTATION", options.permutation);
    options.scaling = env_int("KADATH_MUMPS_SCALING", options.scaling);
    options.residual_tolerance =
        env_double("KADATH_MUMPS_RESIDUAL_TOL", options.residual_tolerance);
    options.pivot_threshold =
        env_double("KADATH_MUMPS_PIVOT_THRESHOLD", options.pivot_threshold);
    options.reuse_analysis = env_int("KADATH_MUMPS_REUSE_ANALYSIS", 1) != 0;
    // "i:value,i:value"
    std::string const extra = env_string("KADATH_MUMPS_ICNTL", "");
    std::size_t pos = 0;
    while (pos < extra.size()) {
        std::size_t const end = std::min(extra.find(',', pos), extra.size());
        std::string const item = extra.substr(pos, end - pos);
        std::size_t const colon = item.find(':');
        if (colon != std::string::npos)
            options.extra_icntl.emplace_back(
                std::atoi(item.substr(0, colon).c_str()),
                std::atoi(item.substr(colon + 1).c_str()));
        pos = end + 1;
    }
    return options;
}

bool MumpsOptions::same_analysis(MumpsOptions const& other) const {
    return centralized == other.centralized && ordering == other.ordering &&
           parallel_analysis == other.parallel_analysis &&
           permutation == other.permutation && scaling == other.scaling &&
           blr_tolerance == other.blr_tolerance &&
           out_of_core == other.out_of_core &&
           omp_threads == other.omp_threads && extra_icntl == other.extra_icntl;
}

MumpsSolver::MumpsSolver(MPI_Comm comm, MumpsOptions const& options)
    : comm_(comm), options_(options) {
    MPI_Comm_rank(comm_, &rank_);
    id_ = DMUMPS_STRUC_C{};
    id_.comm_fortran = static_cast<MUMPS_INT>(MPI_Comm_c2f(comm_));
    id_.par = 1;  // the host also works
    id_.sym = 0;  // unsymmetric
    run(kJobInit);
    if (infog(1) < 0)
        throw std::runtime_error("MUMPS initialization failed: " +
                                 mumps_error_message(infog(1)));

    bool const talk = options_.verbosity > 0;
    set_icntl(1, talk ? 6 : -1);
    set_icntl(2, options_.verbosity > 2 ? 6 : -1);
    set_icntl(3, options_.verbosity > 1 ? 6 : -1);
    set_icntl(4, options_.verbosity);
}

MumpsSolver::~MumpsSolver() {
    int finalized = 0;
    MPI_Finalized(&finalized);
    if (!finalized)  // JOB=-2 communicates; after MPI_Finalize the memory is released by the process exit
        run(kJobEnd);
}

void MumpsSolver::run(int job) {
    id_.job = job;
    dmumps_c(&id_);
}

void MumpsSolver::release_factors() {
    if (!analysed_ && !factored_)
        return;
    run(kJobFreeFactors);
    factored_ = false;
}

bool MumpsSolver::adopt_options(MumpsOptions const& options) {
    if (!options_.same_analysis(options))
        return false;
    options_ = options;
    set_icntl(4, options_.verbosity);
    return true;
}

int MumpsSolver::solve(int n,
                       SparseTriplets& local,
                       std::vector<double>& rhs_solution) {
    int nproc;
    MPI_Comm_size(comm_, &nproc);
    stats_ = MumpsStats{};
    factored_ = false;

    if (static_cast<int>(rhs_solution.size()) != n)
        throw std::invalid_argument(
            "MumpsSolver::solve: rhs size differs from the matrix order");

    // b as given on rank 0, kept on every rank for the residual check of the solution.
    std::vector<double> b = rhs_solution;
    MPI_Bcast(b.data(), n, MPI_DOUBLE, 0, comm_);

    // Kept analysis: bring the matrix to the analysed pattern (explicit zeros) and check that nothing is new.
    // The local pattern decides on every rank; the gathered (centralized) pattern is then the same as well.
    bool keep_analysis = false;
    if (options_.reuse_analysis && analysed_ && analysed_n_ == n) {
        double const start = MPI_Wtime();
        long long counts[3] =
            {0, 0, 0};  // entries outside, zeros added, unsorted ranks
        SparseTriplets merged;
        if (sorted_by_column(local)) {
            std::int64_t zeros = 0;
            counts[0] = merge_with_pattern(local,
                                           pattern_rows_,
                                           pattern_cols_,
                                           merged,
                                           zeros);
            counts[1] = zeros;
        } else
            counts[2] = 1;
        MPI_Allreduce(MPI_IN_PLACE, counts, 3, MPI_LONG_LONG, MPI_SUM, comm_);
        keep_analysis = counts[0] == 0 && counts[2] == 0;
        // Entries that were exact zeros at an earlier step appear later: analysing the union lets the pattern
        // settle after a few steps. ParMETIS analyses the matrix as given (it failed with SIGFPE on a union in
        // test_mumps_wrapper on 4 ranks).
        if (keep_analysis || (counts[2] == 0 && !options_.parallel_analysis))
            local = std::move(merged);
        else
            counts[1] = 0;
        stats_.pattern_zeros = counts[1];
        stats_.time_pattern = MPI_Wtime() - start;
        if (rank_ == 0 && !keep_analysis && options_.verbosity > 0)
            std::cout << "MUMPS: new analysis, "
                      << (counts[2] ? std::string("unsorted entries")
                                    : std::to_string(counts[0]) +
                                          " entries outside the analysed "
                                          "pattern")
                      << std::endl;
    }

    std::int64_t const local_nnz = local.size();
    std::vector<std::int64_t> counts(nproc, 0);
    MPI_Gather(&local_nnz,
               1,
               MPI_INT64_T,
               counts.data(),
               1,
               MPI_INT64_T,
               0,
               comm_);
    MPI_Allreduce(&local_nnz, &stats_.nnz, 1, MPI_INT64_T, MPI_SUM, comm_);

    id_.n = n;
    SparseTriplets global;
    if (options_.centralized) {
        double const start = MPI_Wtime();
        gather_on_root(local.rows, counts, global.rows, comm_);
        gather_on_root(local.cols, counts, global.cols, comm_);
        gather_on_root(local.values, counts, global.values, comm_);
        stats_.time_gather = MPI_Wtime() - start;

        set_icntl(5, 0);
        set_icntl(18, 0);
        set_icntl(6, options_.permutation);
        if (rank_ == 0) {
            id_.nnz = global.size();
            id_.irn = global.rows.data();
            id_.jcn = global.cols.data();
            id_.a = global.values.data();
        }
    } else {
        set_icntl(5, 0);
        set_icntl(18, 3);
        id_.nnz_loc = local_nnz;
        id_.irn_loc = local.rows.data();
        id_.jcn_loc = local.cols.data();
        id_.a_loc = local.values.data();
    }

    set_icntl(8,
              options_.scaling >= 0 ? options_.scaling
                                    : (options_.centralized ? 0 : 77));
    if (options_.parallel_analysis) {
        set_icntl(28, 2);
        set_icntl(29, 2);
    } else {
        set_icntl(28, 1);
        set_icntl(7, options_.ordering);
    }

    set_icntl(20, 0);  // dense centralized right-hand side
    set_icntl(21, 0);  // centralized solution, overwrites rhs on the host
    id_.nrhs = 1;
    id_.lrhs = n;

    if (options_.omp_threads > 0)
        set_icntl(16, options_.omp_threads);
    if (options_.blr_tolerance > 0.) {
        set_icntl(35, 2);
        id_.cntl[7 - 1] = options_.blr_tolerance;
    }
    set_icntl(22, options_.out_of_core ? 1 : 0);
    id_.cntl[1 - 1] = options_.pivot_threshold;
    for (auto const& [i, value] : options_.extra_icntl)
        if (i >= 1 && i <= 60)
            set_icntl(i, value);

    int status = analyse_factor_solve(keep_analysis, n, local, b, rhs_solution);
    if (keep_analysis && status != 0) {
        // The kept ordering may not suit the new values (more delayed pivots, a worse solution): analyse again.
        if (rank_ == 0)
            std::cout << "MUMPS: factorization with the kept analysis failed ("
                      << mumps_error_message(status)
                      << "), repeating it with a new analysis" << std::endl;
        rhs_solution = b;
        status = analyse_factor_solve(false, n, local, b, rhs_solution);
    }

    // The matrix arrays are the caller's or local to this call; JOB=3 only needs the factors.
    id_.irn = id_.jcn = id_.irn_loc = id_.jcn_loc = nullptr;
    id_.a = id_.a_loc = id_.rhs = nullptr;
    if (status != 0)
        return status;
    factored_ = true;
    n_ = n;
    return 0;
}

int MumpsSolver::analyse_factor_solve(bool keep_analysis,
                                      int n,
                                      SparseTriplets const& local,
                                      std::vector<double> const& b,
                                      std::vector<double>& rhs_solution) {
    if (rank_ == 0)
        id_.rhs = rhs_solution.data();

    stats_.analysis_reused = keep_analysis;
    if (!keep_analysis) {
        analysed_ = false;
        double const start = MPI_Wtime();
        run(kJobAnalysis);
        stats_.time_analysis += MPI_Wtime() - start;
        if (infog(1) < 0)
            return infog(1);
        analysis_estimate_mb_ = std::max(1, infog(16));
        analysis_factor_entries_ = millions_if_negative(infog(20));
        if (options_.reuse_analysis) {
            analysed_ = true;
            analysed_n_ = n;
            pattern_rows_ = local.rows;
            pattern_cols_ = local.cols;
        }
    }
    stats_.estimated_factor_entries = analysis_factor_entries_;

    // ICNTL(23): the same explicit working-memory budget on every rank. A rank whose analysis estimate is tiny
    // can receive delayed pivots during factorization; a percentage relaxation (ICNTL(14)) of a tiny estimate
    // stays too small (INFOG(1) = -9 on a BH Jacobian), a budget from the largest estimate does not.
    long const share_mb = node_memory_share_mb(comm_);
    long const cap_mb = share_mb > 0 ? static_cast<long>(0.8 * share_mb)
                                     : LONG_MAX;
    long const estimate_mb =
        analysis_estimate_mb_;  // largest per-rank in-core estimate
    long budget_mb = options_.max_mem_mb > 0
                         ? options_.max_mem_mb
                         : estimate_mb + estimate_mb / 4 +
                               64;  // ~1.25x: delayed pivots fit, see header
    if (options_.max_mem_mb <=
        0)  // the cap limits only the margin, never below the estimate itself
        budget_mb = std::max(estimate_mb, std::min(budget_mb, cap_mb));

    int mem_relax = options_.mem_relax_percent;
    double start = MPI_Wtime();
    for (int attempt = 0;; ++attempt) {
        set_icntl(14, mem_relax);
        set_icntl(23, static_cast<int>(std::min<long>(budget_mb, INT_MAX)));
        run(kJobFactorSolve);
        if (!is_workspace_error(infog(1)) ||
            attempt == options_.max_mem_retries)
            break;
        long const next_mb =
            std::max(budget_mb, std::min(2 * budget_mb, cap_mb));
        if (rank_ == 0)
            std::cout << "MUMPS: workspace too small (INFOG(1)=" << infog(1)
                      << ") with ICNTL(23)=" << budget_mb
                      << " MB, retrying with ICNTL(23)=" << next_mb
                      << " MB, ICNTL(14)=" << 2 * mem_relax << std::endl;
        if (next_mb <= budget_mb && rank_ == 0)
            std::cout << "MUMPS: memory budget already at 80 % of the free "
                         "node memory per rank"
                      << std::endl;
        budget_mb = next_mb;
        mem_relax *= 2;
    }
    stats_.time_factor_solve += MPI_Wtime() - start;
    stats_.mem_relax_used = mem_relax;
    stats_.mem_budget_mb = budget_mb;
    stats_.off_diagonal_pivots = infog(12);
    stats_.delayed_pivots = infog(13);
    if (infog(1) < 0)
        return infog(1);

    stats_.factor_entries = millions_if_negative(infog(29));
    stats_.factor_flops = id_.rinfog[3 - 1];
    stats_.peak_mem_mb_max = infog(21);
    stats_.peak_mem_mb_total = infog(22);
    stats_.ordering_used = infog(7);

    MPI_Bcast(rhs_solution.data(), n, MPI_DOUBLE, 0, comm_);

    // Residual check with the entries as given (MUMPS does not modify them): every rank multiplies its own.
    std::int64_t const local_nnz = local.size();
    std::vector<double> local_ax(n, 0.), ax(n, 0.);
    for (std::int64_t k = 0; k < local_nnz; ++k)
        local_ax[local.rows[k] - 1] += local.values[k] *
                                       rhs_solution[local.cols[k] - 1];
    MPI_Allreduce(local_ax.data(), ax.data(), n, MPI_DOUBLE, MPI_SUM, comm_);
    double diff = 0., scale = 0.;
    for (int i = 0; i < n; ++i) {
        diff = std::max(diff, std::fabs(ax[i] - b[i]));
        scale = std::max(scale, std::fabs(b[i]));
    }
    stats_.relative_residual = scale > 0. ? diff / scale : diff;
    if (!(stats_.relative_residual <= options_.residual_tolerance))
        return kInaccurateSolution;
    return 0;
}

int MumpsSolver::solve_factored(std::vector<double>& rhs_solution) {
    if (!factored_ || static_cast<int>(rhs_solution.size()) != n_)
        return -1;
    set_icntl(20, 0);
    set_icntl(21, 0);
    id_.nrhs = 1;
    id_.lrhs = n_;
    id_.rhs = rank_ == 0 ? rhs_solution.data() : nullptr;
    run(kJobSolve);
    id_.rhs = nullptr;
    if (infog(1) < 0)
        return infog(1);
    MPI_Bcast(rhs_solution.data(), n_, MPI_DOUBLE, 0, comm_);
    return 0;
}

std::vector<MumpsAttempt> mumps_retry_ladder(MumpsOptions const& base) {
    std::vector<MumpsAttempt> ladder = {{"configured options", base}};
    if (!base.retry)
        return ladder;
    MumpsOptions recovery = base;
    recovery.scaling = -1;
    recovery.max_mem_mb = 0;
    recovery.max_mem_retries = MumpsOptions{}.max_mem_retries;

    MumpsOptions structural = recovery;
    structural.centralized = true;
    structural.permutation = 1;
    ladder.push_back({"structural permutation ICNTL(6)=1", structural});

    MumpsOptions distributed = recovery;
    distributed.centralized = false;
    ladder.push_back({"distributed input", distributed});
    return ladder;
}

std::string mumps_error_message(int infog1) {
    switch (infog1) {
        case -5:
        case -13:
            return "memory allocation failed (" + std::to_string(infog1) + ")";
        case -6:
            return "matrix is structurally singular (-6)";
        case -8:
        case -9:
        case -17:
        case -20:
            return "workspace too small even after retries (" +
                   std::to_string(infog1) + "), raise KADATH_MUMPS_MEM_RELAX";
        case -10:
            return "matrix is numerically singular (-10)";
        case kInaccurateSolution:
            return "MUMPS reported success but the solution fails the residual "
                   "check (max|Ax-b|/max|b| > "
                   "KADATH_MUMPS_RESIDUAL_TOL)";
        case -19:
            return "ICNTL(23) memory budget too small even after retries (-19)";
        default:
            return "MUMPS error INFOG(1)=" + std::to_string(infog1) +
                   " (see MUMPS user guide, section 8)";
    }
}

}  // namespace linear_solvers
}  // namespace Kadath

#endif  // KADATH_USE_MUMPS
