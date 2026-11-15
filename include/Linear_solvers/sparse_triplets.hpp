#ifndef __SPARSE_TRIPLETS_HPP_
#define __SPARSE_TRIPLETS_HPP_

#include <algorithm>
#include <cstdint>
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
