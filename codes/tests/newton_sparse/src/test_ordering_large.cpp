/*
 * T8b: MUMPS analysis of a graph with more than 2^31 entries (heavy; one full node, run separately).
 *
 * The matrix is block diagonal with dense upper-triangular blocks, given as distributed input: the analysis
 * orders the graph of A + A^T, whose adjacency has blocks x b (b - 1) entries, by default about 2.6e9 > 2^31 - 1
 * from 1.3e9 non-zeros. Only the analysis (JOB = 1) runs, without the value-based permutation (ICNTL(6) = 0),
 * so no values are needed. With 32-bit orderings MUMPS stops with INFOG(1) = -51; with 64-bit orderings the
 * analysis must succeed with the requested ordering (INFOG(7), INFOG(7) = 5 for METIS; for the parallel
 * orderings INFOG(7) is not set and ICNTL(28)'s outcome is checked through INFOG(32)).
 *
 * Status (MUMPS 5.8.1, 2026-11-15): sequential METIS passes above 2^31 (2.6e9, 34 s); the parallel orderings pass
 * below (1.6e9) but fail above it (ParMETIS: Intel MPI abort; PT-Scotch: heap corruption after the ordering),
 * so MumpsSolver uses the sequential analysis for such graphs.
 *
 * usage: mpirun -np <p> test_ordering_large [metis|parmetis|ptscotch] [block size] [blocks]
 * Memory: about 8 bytes per non-zero over all ranks for the input, plus the gathered graph on rank 0 for the
 * sequential ordering (about 16 bytes per adjacency entry, ~40 GB at the default size).
 * Returns 0 if the analysis succeeds with the requested ordering, 1 otherwise, 77 when built without MUMPS.
 */
#include <mpi.h>

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#ifdef KADATH_USE_MUMPS
#include "dmumps_c.h"
#endif

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, nproc;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
#ifndef KADATH_USE_MUMPS
    if (rank == 0)
        std::cout << "test_ordering_large: SKIP (built without MUMPS)" << std::endl;
    MPI_Finalize();
    return 77;
#else
    std::string const ordering = argc > 1 ? argv[1] : "metis";
    std::int64_t const b = argc > 2 ? std::atoll(argv[2]) : 2000;
    std::int64_t const blocks = argc > 3 ? std::atoll(argv[3]) : 650;
    std::int64_t const n = b * blocks;
    std::int64_t const adjacency = blocks * b * (b - 1);
    if (rank == 0)
        std::cout << "test_ordering_large: ordering " << ordering << ", n = " << n << ", nnz = " << blocks * b * (b + 1) / 2
                  << ", adjacency of A + A^T = " << adjacency << " (2^31 - 1 = 2147483647)" << std::endl;

    // Blocks dealt round-robin to the ranks; entries (i, j), i <= j, of each dense upper-triangular block.
    std::vector<MUMPS_INT> irn, jcn;
    for (std::int64_t blk = rank; blk < blocks; blk += nproc)
        for (std::int64_t j = 0; j < b; ++j)
            for (std::int64_t i = 0; i <= j; ++i) {
                irn.push_back(static_cast<MUMPS_INT>(blk * b + i + 1));
                jcn.push_back(static_cast<MUMPS_INT>(blk * b + j + 1));
            }

    DMUMPS_STRUC_C id{};
    id.comm_fortran = static_cast<MUMPS_INT>(MPI_Comm_c2f(MPI_COMM_WORLD));
    id.par = 1;
    id.sym = 0;
    id.job = -1;
    dmumps_c(&id);
    id.icntl[0] = 6;
    id.icntl[1] = -1;
    id.icntl[2] = 6;
    id.icntl[3] = 2;
    id.icntl[5] = 0;   // ICNTL(6): no value-based permutation (no values given)
    id.icntl[17] = 3;  // ICNTL(18): distributed assembled input
    if (ordering == "metis") {
        id.icntl[27] = 1;  // ICNTL(28): sequential analysis
        id.icntl[6] = 5;   // ICNTL(7): METIS
    } else {
        id.icntl[27] = 2;                                // ICNTL(28): parallel analysis
        id.icntl[28] = ordering == "ptscotch" ? 1 : 2;  // ICNTL(29): PT-Scotch / ParMETIS
    }
    id.n = static_cast<MUMPS_INT>(n);
    id.nnz_loc = static_cast<MUMPS_INT8>(irn.size());
    id.irn_loc = irn.data();
    id.jcn_loc = jcn.data();
    double const start = MPI_Wtime();
    id.job = 1;
    dmumps_c(&id);
    double const elapsed = MPI_Wtime() - start;

    // Peak resident memory (VmHWM) of rank 0, which holds the gathered graph of the sequential analysis, and the
    // largest of the other ranks.
    long hwm_kb = 0;
    {
        std::ifstream status("/proc/self/status");
        std::string key;
        while (status >> key)
            if (key == "VmHWM:") {
                status >> hwm_kb;
                break;
            }
    }
    long other_kb = rank == 0 ? 0 : hwm_kb, other_max_kb = 0;
    MPI_Reduce(&other_kb, &other_max_kb, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    int failures = 0;
    if (rank == 0) {
        std::cout << "memory: rank 0 VmHWM " << hwm_kb / 1048576. << " GB, other ranks max " << other_max_kb / 1048576.
                  << " GB" << std::endl;
        int const infog1 = id.infog[0], infog2 = id.infog[1], infog7 = id.infog[6], infog32 = id.infog[31];
        std::cout << "analysis: INFOG(1) = " << infog1 << ", INFOG(2) = " << infog2 << ", INFOG(7) = " << infog7
                  << ", INFOG(32) = " << infog32 << ", estimated factor entries INFOG(20) = " << id.infog[19]
                  << ", " << elapsed << " s" << std::endl;
        bool const ok = infog1 >= 0;
        failures += !ok;
        std::cout << (ok ? "ok:   " : "FAIL: ") << "analysis of a graph with " << adjacency << " entries"
                  << (infog1 == -51 ? " (INFOG(1) = -51: the ordering uses 32-bit indices)" : "") << std::endl;
        if (ok && ordering == "metis") {
            bool const used = infog7 == 5;
            failures += !used;
            std::cout << (used ? "ok:   " : "FAIL: ") << "METIS was the ordering used (INFOG(7) = " << infog7 << ")"
                      << std::endl;
        }
        if (ok && ordering != "metis") {
            bool const parallel = infog32 == 2;
            failures += !parallel;
            std::cout << (parallel ? "ok:   " : "FAIL: ") << "the analysis ran in parallel (INFOG(32) = " << infog32
                      << ")" << std::endl;
        }
        std::cout << "test_ordering_large: " << (failures == 0 ? "PASS" : "FAIL") << " (" << failures << " failures)"
                  << std::endl;
    }
    id.job = -2;
    dmumps_c(&id);
    MPI_Bcast(&failures, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return failures == 0 ? 0 : 1;
#endif
}
