#ifdef KADATH_USE_MUMPS

#include "Linear_solvers/mumps_solver.hpp"
#include "Linear_solvers/weighted_matching.hpp"

#include <algorithm>
#include <climits>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace Kadath {
namespace linear_solvers {

namespace {

constexpr int kJobInit = -1;
constexpr int kJobEnd = -2;
constexpr int kJobAnalysis = 1;
constexpr int kJobSolve = 3;
constexpr int kJobFactorSolve = 5;

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

/// Column permutation for distributed input (the ICNTL(6) maximum-product matching, which MUMPS only
/// computes for a matrix centralized on rank 0). Every rank keeps, of each of its columns, the @p per_column
/// largest entries and the @p per_column largest entries relative to their row maxima; only those are
/// gathered on rank 0 and matched there. If they hold no perfect matching, rank 0 asks for all entries of
/// the columns its failed searches reached (the only entries that can complete it) and matches again, for
/// at most @p max_rounds rounds and while the gathered entries stay below @p max_fraction of the matrix.
/// Returns, on every rank, the original column (1-based) of permuted column i, so that row i holds a large
/// entry on the diagonal; empty if no perfect matching was found. Collective.
std::vector<int> matching_permutation(int n,
                                      SparseTriplets const& local,
                                      int per_column,
                                      int max_rounds,
                                      double max_fraction,
                                      MPI_Comm comm) {
    int rank, nproc;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nproc);

    std::vector<double> rowmax(n, 0.);
    for (std::int64_t k = 0; k < local.size(); ++k)
        rowmax[local.rows[k] - 1] =
            std::max(rowmax[local.rows[k] - 1], std::fabs(local.values[k]));
    MPI_Allreduce(MPI_IN_PLACE, rowmax.data(), n, MPI_DOUBLE, MPI_MAX, comm);

    // Local entries grouped by column.
    std::vector<std::int64_t> order;
    order.reserve(local.size());
    for (std::int64_t k = 0; k < local.size(); ++k)
        if (local.values[k] != 0.)
            order.push_back(k);
    std::sort(order.begin(), order.end(), [&](std::int64_t a, std::int64_t b) {
        return local.cols[a] < local.cols[b];
    });
    std::vector<char> keep(local.size(),
                           0);  // 1 = selected now, 2 = already sent
    auto keep_largest = [&](auto first, auto last, auto key) {
        if (last - first > per_column)
            std::nth_element(first,
                             first + per_column,
                             last,
                             [&](std::int64_t a, std::int64_t b) {
                                 return key(a) > key(b);
                             });
        for (auto it = first; it != last && it - first < per_column; ++it)
            keep[*it] = 1;
    };
    for (auto first = order.begin(); first != order.end();) {
        int const col = local.cols[*first];
        auto last = first;
        while (last != order.end() && local.cols[*last] == col)
            ++last;
        keep_largest(first, last, [&](std::int64_t k) {
            return std::fabs(local.values[k]);
        });
        keep_largest(first, last, [&](std::int64_t k) {
            return std::fabs(local.values[k]) / rowmax[local.rows[k] - 1];
        });
        first = last;
    }
    std::vector<std::int64_t>().swap(order);
    std::vector<double>().swap(rowmax);

    std::int64_t total_nnz = 0, sent = 0;
    std::int64_t const local_nnz = local.size();
    MPI_Allreduce(&local_nnz, &total_nnz, 1, MPI_INT64_T, MPI_SUM, comm);

    SparseTriplets global;  // rank 0: everything gathered so far
    std::vector<int> perm(n, 0);
    for (int round = 0;; ++round) {
        // Send the newly selected entries.
        SparseTriplets fresh;
        for (std::int64_t k = 0; k < local.size(); ++k)
            if (keep[k] == 1) {
                keep[k] = 2;
                fresh.rows.push_back(local.rows[k]);
                fresh.cols.push_back(local.cols[k]);
                fresh.values.push_back(local.values[k]);
            }
        std::int64_t const fresh_nnz = fresh.size();
        std::vector<std::int64_t> counts(nproc, 0);
        MPI_Gather(&fresh_nnz,
                   1,
                   MPI_INT64_T,
                   counts.data(),
                   1,
                   MPI_INT64_T,
                   0,
                   comm);
        SparseTriplets received;
        gather_on_root(fresh.rows, counts, received.rows, comm);
        gather_on_root(fresh.cols, counts, received.cols, comm);
        gather_on_root(fresh.values, counts, received.values, comm);
        global.rows.insert(global.rows.end(),
                           received.rows.begin(),
                           received.rows.end());
        global.cols.insert(global.cols.end(),
                           received.cols.begin(),
                           received.cols.end());
        global.values.insert(global.values.end(),
                             received.values.begin(),
                             received.values.end());

        std::vector<int> stuck;
        int found = 0;
        if (rank == 0) {
            sent = global.size();
            std::vector<std::int64_t> ptr(n + 1, 0);
            for (int c : global.cols)
                ++ptr[c];
            for (int j = 0; j < n; ++j)
                ptr[j + 1] += ptr[j];
            std::vector<std::int64_t> next(ptr.begin(), ptr.end() - 1);
            std::vector<int> row(global.size());
            std::vector<double> val(global.size());
            for (std::int64_t k = 0; k < global.size(); ++k) {
                std::int64_t const at = next[global.cols[k] - 1]++;
                row[at] = global.rows[k] - 1;
                val[at] = global.values[k];
            }
            std::vector<int> const row_of_col =
                max_product_matching(n, ptr, row, val, &stuck);
            if (!row_of_col.empty()) {
                found = 1;
                if (round > 0)
                    std::cout
                        << "MUMPS: matching round " << round
                        << ": perfect matching on " << sent << " entries ("
                        << 100. * sent / std::max<std::int64_t>(1, total_nnz)
                        << " %)" << std::endl;
                for (int j = 0; j < n; ++j)
                    perm[row_of_col[j]] = j + 1;
            } else {
                std::cout << "MUMPS: matching round " << round << ": " << sent
                          << " entries ("
                          << 100. * sent / std::max<std::int64_t>(1, total_nnz)
                          << " %), " << stuck.size()
                          << " columns reached by failed searches" << std::endl;
            }
        }
        MPI_Bcast(&found, 1, MPI_INT, 0, comm);
        if (found) {
            MPI_Bcast(perm.data(), n, MPI_INT, 0, comm);
            return perm;
        }
        MPI_Bcast(&sent, 1, MPI_INT64_T, 0, comm);
        int stuck_n = static_cast<int>(stuck.size());
        MPI_Bcast(&stuck_n, 1, MPI_INT, 0, comm);
        if (round + 1 >= max_rounds || stuck_n == 0 ||
            sent >= max_fraction * total_nnz)
            return {};
        stuck.resize(stuck_n);
        MPI_Bcast(stuck.data(), stuck_n, MPI_INT, 0, comm);
        std::vector<char> expand(n, 0);
        for (int j : stuck)
            expand[j] = 1;
        for (std::int64_t k = 0; k < local.size(); ++k)
            if (keep[k] == 0 && local.values[k] != 0. &&
                expand[local.cols[k] - 1])
                keep[k] = 1;
    }
}

}  // namespace

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

    // Distributed input: MUMPS cannot compute the ICNTL(6) column permutation, so it is computed here and
    // applied to the local columns; the solution is permuted back below.
    col_perm_.clear();
    if (!options_.centralized && options_.matching_entries > 0) {
        double const start = MPI_Wtime();
        col_perm_ = matching_permutation(n,
                                         local,
                                         options_.matching_entries,
                                         8,
                                         0.25,
                                         comm_);
        stats_.time_matching = MPI_Wtime() - start;
        if (col_perm_.empty())
            return kNoMatching;
        std::vector<int> inv(n);
        for (int i = 0; i < n; ++i)
            inv[col_perm_[i] - 1] = i + 1;
        for (auto& c : local.cols)
            c = inv[c - 1];
    }

    struct
        RestoreColumns {  // the caller's triplets keep their original columns on every exit path
        std::vector<int> const& perm;
        SparseTriplets& t;
        bool done = false;

        ~RestoreColumns() {
            if (!done && !perm.empty())
                for (auto& c : t.cols)
                    c = perm[c - 1];
        }
    } restore_columns{col_perm_, local};

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
    // MUMPS's parallel analysis fails on graphs above 2^31 entries even with 64-bit ParMETIS/PT-Scotch
    // (ParMETIS: abort in Intel MPI; PT-Scotch: heap corruption after the ordering; test_ordering_large),
    // while the sequential METIS ordering handles them. The graph of A + A^T has at most 2 nnz entries.
    bool const graph_fits_parallel = 2 * stats_.nnz <
                                     std::numeric_limits<std::int32_t>::max();
    if (options_.parallel_analysis && options_.ordering == 5 &&
        n >= options_.parallel_analysis_min_n && graph_fits_parallel) {
        set_icntl(28, 2);
        set_icntl(29, options_.parallel_ordering);
    } else {
        set_icntl(28, 1);
        set_icntl(7, options_.ordering);
    }

    set_icntl(20, 0);  // dense centralized right-hand side
    set_icntl(21, 0);  // centralized solution, overwrites rhs on the host
    id_.nrhs = 1;
    id_.lrhs = n;
    if (rank_ == 0)
        id_.rhs = rhs_solution.data();
    set_icntl(16, 1);  // one OpenMP thread per MPI rank
    id_.cntl[1 - 1] = options_.pivot_threshold;
    // Up to 3 steps of iterative refinement with the factors (stops when converged): a safeguard for Jacobians
    // with many delayed pivots, at the cost of a few solves.
    set_icntl(10, 3);

    double start = MPI_Wtime();
    run(kJobAnalysis);
    stats_.time_analysis = MPI_Wtime() - start;
    if (infog(1) < 0)
        return infog(1);
    stats_.estimated_factor_entries = millions_if_negative(infog(20));

    // ICNTL(23): the same explicit working-memory budget on every rank. A rank whose analysis estimate is tiny
    // can receive delayed pivots during factorization; a percentage relaxation (ICNTL(14)) of a tiny estimate
    // stays too small (INFOG(1) = -9 on a BH Jacobian), a budget from the largest estimate does not.
    long const share_mb = node_memory_share_mb(comm_);
    long const cap_mb = share_mb > 0 ? static_cast<long>(0.8 * share_mb)
                                     : LONG_MAX;
    long const estimate_mb =
        std::max(1, infog(16));  // largest per-rank in-core estimate
    long budget_mb = options_.max_mem_mb > 0
                         ? options_.max_mem_mb
                         : estimate_mb + estimate_mb / 4 +
                               64;  // ~1.25x: delayed pivots fit, see header
    if (options_.max_mem_mb <=
        0)  // the cap limits only the margin, never below the estimate itself
        budget_mb = std::max(estimate_mb, std::min(budget_mb, cap_mb));

    int mem_relax = options_.mem_relax_percent;
    start = MPI_Wtime();
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
    stats_.time_factor_solve = MPI_Wtime() - start;
    stats_.mem_budget_mb = budget_mb;
    stats_.off_diagonal_pivots = infog(12);
    stats_.delayed_pivots = infog(13);
    if (infog(1) < 0)
        return infog(1);

    stats_.factor_entries = millions_if_negative(infog(29));
    stats_.factor_flops = id_.rinfog[3 - 1];
    stats_.peak_mem_mb_max = infog(21);
    stats_.peak_mem_mb_total = infog(22);

    MPI_Bcast(rhs_solution.data(), n, MPI_DOUBLE, 0, comm_);
    if (!col_perm_
             .empty()) {  // y of the column-permuted system -> x; the caller's columns are restored
        std::vector<double> x(n);
        for (int i = 0; i < n; ++i)
            x[col_perm_[i] - 1] = rhs_solution[i];
        rhs_solution.swap(x);
        for (auto& c : local.cols)
            c = col_perm_[c - 1];
        restore_columns.done = true;
    }

    // Residual check with the entries as given (MUMPS does not modify them): every rank multiplies its own.
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

    // The matrix arrays are the caller's or local to this call; JOB=3 only needs the factors.
    id_.irn = id_.jcn = id_.irn_loc = id_.jcn_loc = nullptr;
    id_.a = id_.a_loc = id_.rhs = nullptr;
    if (!(stats_.relative_residual <= options_.residual_tolerance))
        return kInaccurateSolution;
    factored_ = true;
    n_ = n;
    return 0;
}

int MumpsSolver::solve_factored(std::vector<double>& rhs_solution) {
    if (!factored_ || static_cast<int>(rhs_solution.size()) != n_)
        return -1;
    set_icntl(20, 0);
    set_icntl(21, 0);
    set_icntl(10, 0);  // no refinement: the matrix is not kept for JOB=3
    id_.nrhs = 1;
    id_.lrhs = n_;
    id_.rhs = rank_ == 0 ? rhs_solution.data() : nullptr;
    run(kJobSolve);
    id_.rhs = nullptr;
    if (infog(1) < 0)
        return infog(1);
    MPI_Bcast(rhs_solution.data(), n_, MPI_DOUBLE, 0, comm_);
    if (!col_perm_.empty()) {
        std::vector<double> x(n_);
        for (int i = 0; i < n_; ++i)
            x[col_perm_[i] - 1] = rhs_solution[i];
        rhs_solution.swap(x);
    }
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

    if (!base.centralized) {  // MUMPS's own permutation on the gathered matrix
        MumpsOptions centralized = recovery;
        centralized.centralized = true;
        ladder.push_back({"centralized input", centralized});
    }

    MumpsOptions structural = recovery;
    structural.centralized = true;
    structural.permutation = 1;
    ladder.push_back({"structural permutation ICNTL(6)=1", structural});

    MumpsOptions distributed = recovery;
    distributed.centralized = false;
    distributed.matching_entries =
        0;  // no gather at all; large fill, but it can still succeed
    ladder.push_back({"distributed input without permutation", distributed});
    return ladder;
}

MumpsOptions with_internal_ordering(MumpsOptions options) {
    options.ordering = 2;  // AMF
    options.parallel_analysis = false;
    return options;
}

std::string mumps_error_message(int infog1) {
    switch (infog1) {
        case -5:
        case -13:
            return "memory allocation failed (" + std::to_string(infog1) + ")";
        case -6:
            return "matrix is structurally singular (-6)";
        case kGraphTooLarge:
            return "graph too large for the 32-bit METIS/ParMETIS ordering "
                   "(-51)";
        case -8:
        case -9:
        case -17:
        case -20:
            return "workspace too small even after retries (" +
                   std::to_string(infog1) + "): run on more nodes";
        case -10:
            return "matrix is numerically singular (-10)";
        case kNoMatching:
            return "no perfect matching for the column permutation "
                   "(structurally singular pattern)";
        case kInaccurateSolution:
            return "MUMPS reported success but the solution fails the residual "
                   "check (max|Ax-b|/max|b| too large)";
        case -19:
            return "memory budget too small even after retries (-19): run on "
                   "more nodes";
        default:
            return "MUMPS error INFOG(1)=" + std::to_string(infog1) +
                   " (see MUMPS user guide, section 8)";
    }
}

}  // namespace linear_solvers
}  // namespace Kadath

#endif  // KADATH_USE_MUMPS
