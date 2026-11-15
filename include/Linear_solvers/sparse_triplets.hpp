#ifndef __SPARSE_TRIPLETS_HPP_
#define __SPARSE_TRIPLETS_HPP_

#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>

#include "array.hpp"

namespace Kadath {
namespace linear_solvers {

/// Locally owned entries of a sparse matrix in 1-based coordinate (COO) format, as MUMPS expects.
struct SparseTriplets {
    std::vector<int> rows;
    std::vector<int> cols;
    std::vector<double> values;

    std::int64_t size() const {
        return static_cast<std::int64_t>(values.size());
    }
};

/// Appends the entries of column @p global_col (0-based) to @p triplets, dropping exact zeros.
/// Exact zeros are the structural zeros left by do_col_J for equations outside the perturbed domain.
inline void append_column(Array<double> const& column,
                          int global_col,
                          SparseTriplets& triplets) {
    int const nrows = column.get_size(0);
    for (int i = 0; i < nrows; ++i) {
        double const value = column(i);
        if (value != 0.) {
            triplets.rows.push_back(i + 1);
            triplets.cols.push_back(global_col + 1);
            triplets.values.push_back(value);
        }
    }
}

/// True when the entries are sorted by column and then row without duplicates (as append_column builds them).
inline bool sorted_by_column(SparseTriplets const& t) {
    for (std::int64_t k = 1; k < t.size(); ++k)
        if (t.cols[k] < t.cols[k - 1] ||
            (t.cols[k] == t.cols[k - 1] && t.rows[k] <= t.rows[k - 1]))
            return false;
    return true;
}

/// Writes to @p u the union of the entries of @p t and the pattern (@p rows, @p cols), both sorted by column and
/// then row: pattern positions missing from @p t become explicit zeros. @return the number of entries of @p t
/// outside the pattern (0: the union is the pattern); @p zeros receives the number of zeros added.
inline std::int64_t merge_with_pattern(SparseTriplets const& t,
                                       std::vector<int> const& rows,
                                       std::vector<int> const& cols,
                                       SparseTriplets& u,
                                       std::int64_t& zeros) {
    std::int64_t const np = static_cast<std::int64_t>(rows.size());
    u = SparseTriplets{};
    u.rows.reserve(std::max<std::int64_t>(np, t.size()));
    u.cols.reserve(u.rows.capacity());
    u.values.reserve(u.rows.capacity());
    std::int64_t i = 0, k = 0, outside = 0;
    zeros = 0;
    while (i < t.size() || k < np) {
        bool take_t, take_p;
        if (i == t.size()) {
            take_t = false;
            take_p = true;
        } else if (k == np) {
            take_t = true;
            take_p = false;
        } else {
            std::int64_t const a =
                (static_cast<std::int64_t>(t.cols[i]) << 32) | t.rows[i];
            std::int64_t const b = (static_cast<std::int64_t>(cols[k]) << 32) |
                                   rows[k];
            take_t = a <= b;
            take_p = b <= a;
        }
        if (take_t) {
            u.rows.push_back(t.rows[i]);
            u.cols.push_back(t.cols[i]);
            u.values.push_back(t.values[i]);
            if (!take_p)
                ++outside;
            ++i;
        } else {
            u.rows.push_back(rows[k]);
            u.cols.push_back(cols[k]);
            u.values.push_back(0.);
            ++zeros;
        }
        if (take_p)
            ++k;
    }
    return outside;
}

/// Largest |value| per row and per column (0-based, length n) of the local entries; the caller reduces the
/// row maxima across ranks (columns are never split across ranks).
inline void local_row_column_max(SparseTriplets const& triplets,
                                 int n,
                                 std::vector<double>& row_max,
                                 std::vector<double>& col_max) {
    row_max.assign(n, 0.);
    col_max.assign(n, 0.);
    for (std::int64_t k = 0; k < triplets.size(); ++k) {
        double const a = triplets.values[k] < 0. ? -triplets.values[k]
                                                 : triplets.values[k];
        double& r = row_max[triplets.rows[k] - 1];
        double& c = col_max[triplets.cols[k] - 1];
        r = std::max(r, a);
        c = std::max(c, a);
    }
}

/// Removes entries negligible in both directions: |a_ij| <= tolerance * max_k |a_ik| and
/// |a_ij| <= tolerance * max_k |a_kj|. Only for preconditioners: the matrix changes.
/// @return number of entries removed.
inline std::int64_t drop_negligible(SparseTriplets& triplets,
                                    std::vector<double> const& row_max,
                                    std::vector<double> const& col_max,
                                    double tolerance) {
    std::int64_t kept = 0;
    std::int64_t const total = triplets.size();
    for (std::int64_t k = 0; k < total; ++k) {
        double const a = triplets.values[k] < 0. ? -triplets.values[k]
                                                 : triplets.values[k];
        if (a <= tolerance * row_max[triplets.rows[k] - 1] &&
            a <= tolerance * col_max[triplets.cols[k] - 1])
            continue;
        triplets.rows[kept] = triplets.rows[k];
        triplets.cols[kept] = triplets.cols[k];
        triplets.values[kept] = triplets.values[k];
        ++kept;
    }
    triplets.rows.resize(kept);
    triplets.cols.resize(kept);
    triplets.values.resize(kept);
    triplets.rows.shrink_to_fit();
    triplets.cols.shrink_to_fit();
    triplets.values.shrink_to_fit();
    return total - kept;
}

/// Block size of the 1-D column distribution of the Newton Jacobian (same rule as the ScaLAPACK path).
/// @param m number of rows, @param nproc number of ranks; the result is at least 1.
inline int newton_block_size(int m, int nproc) {
    int bsize = 256;
    while (bsize > 1 && nproc * bsize > m)
        bsize /= 2;
    return bsize;
}

/// 0-based global columns owned by @p rank in a 1-D block-cyclic distribution of @p n columns.
/// Matches ScaLAPACK numroc/indxl2g on a 1 x nproc process grid.
inline std::vector<int> block_cyclic_columns(int n,
                                             int bsize,
                                             int rank,
                                             int nproc) {
    std::vector<int> cols;
    for (int start = rank * bsize; start < n; start += nproc * bsize)
        for (int j = start; j < std::min(start + bsize, n); ++j)
            cols.push_back(j);
    return cols;
}

}  // namespace linear_solvers
}  // namespace Kadath

#endif
