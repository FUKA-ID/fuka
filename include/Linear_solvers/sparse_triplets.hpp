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
