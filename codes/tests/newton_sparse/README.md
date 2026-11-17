# newton_sparse tests

Unit and integration tests of the sparse Newton solvers (`src_par/do_newton.cpp`, `src/Linear_solvers/`,
the Jacobian shortcuts in `src/System_of_eqs/solver.cpp` and `src/Ope_eq/ope_eq.cpp`).
Each test is a standalone executable that returns 0 on success, 1 on failure and 77 when
the library was built without MUMPS (`KADATH_USE_MUMPS` undefined).

| Test | Ranks | Checks |
|---|---|---|
| `test_sparse_triplets` | serial | block size rule, 1-D block-cyclic column ownership vs ScaLAPACK `numroc`/`indxl2g`, column to COO round trip (only exact zeros dropped), row/column maxima and `drop_negligible` |
| `test_gmres` | serial | right-preconditioned GMRES on dense systems: requested tolerance met by the true residual, one iteration with the exact preconditioner, few with a stale one, iteration cap, zero right-hand side |
| `test_weighted_matching` | serial | maximum-product matching used for the column permutation of distributed input: optimal against brute force (n <= 8), recovers a permuted diagonally dominant matrix, structurally singular patterns and the columns reported by failed searches, the warm start from a previous matching and its potentials (optimal like a cold start), n = 2000 |
| `test_ordering_int64` | serial | MUMPS was built against METIS/ParMETIS (and Scotch, if enabled) with 64-bit indices, as reported by MUMPS itself (`mumps_metis_idxsize_`, `mumps_scotch_intsize_`); with 32-bit orderings large Jacobians fail with `INFOG(1) = -51` and fall back to the slower AMF ordering |
| `test_ordering_large` | 40 (one node) | heavy, run separately: MUMPS analysis of a block matrix whose A + A^T graph has 2.6e9 > 2^31 entries (default size) with the sequential `metis` ordering, about 35 s; `parmetis` and `ptscotch` (argument) pass below 2^31 (`2000 400`, 1.6e9) but MUMPS 5.8.1's parallel analysis fails above it, so the solver uses the sequential analysis there |
| `test_mumps_wrapper` | 1, 2, 4 | `MumpsSolver` on a block-sparse matrix with bordering rows/columns and a zero diagonal: residual, error, identical solution on all ranks, instance reuse, centralized/distributed input, distributed input with the wrapper's column permutation (per-column and per-row candidates, gradual growth of the stuck columns with and without warm start, the hint from an earlier permutation: perfect in the first round for a perturbed matrix, a stale or wrong-sized hint harmless), ParMETIS, AMF ordering (the fallback after INFOG(1) = -51), independence of the column distribution, residual-check rejection, retry ladder, out-of-core on demand and the insufficient-memory stop (`KADATH_MUMPS_OOC=off`), block low-rank factors, singular matrix detection |
| `test_newton_kadath` | 1, 4 | a nonlinear Kadath system with a `var_double` and an `eq_int` row (analytic solution `P = log(1 + a/r)`, `lam = 1/2`): `do_col_J` determinism (T3), `do_JX(e_j) == do_col_J(j)` incl. adapted and homothetic variable domains (T3b, T3c), the `do_col_J` / `do_JX` shortcuts against the full evaluation (T3d, T3e), one Newton update MUMPS vs ScaLAPACK (T4a), full solves with MUMPS and ScaLAPACK (T4b) and Newton-Krylov (T4d) against each other and the analytic solution, the divergence criterion that stops the MUMPS fallbacks (T4f) |

## Build and run

```bash
# library first, built with MUMPS (see installation.md), then
cd $HOME_KADATH/codes/tests/newton_sparse && . compile
export OMP_NUM_THREADS=1
./bin/Release/test_sparse_triplets
./bin/Release/test_gmres
./bin/Release/test_weighted_matching
./bin/Release/test_ordering_int64
for np in 1 2 4; do mpirun -np $np ./bin/Release/test_mumps_wrapper; done
for np in 1 4; do mpirun -np $np ./bin/Release/test_newton_kadath; done
```

The large-graph check needs a full node (inside a job):

```bash
mpirun -np 40 ./bin/Release/test_ordering_large metis              # graph 2.6e9 > 2^31
for o in parmetis ptscotch; do mpirun -np 40 ./bin/Release/test_ordering_large $o 2000 400; done
```

`test_newton_kadath` takes about 5 minutes on 1 rank and 2 minutes on 4; the others a few seconds.
On a cluster, run them inside a job (e.g. `salloc -n 4`), not on a login node.

## Solver selection

The Newton linear solver is chosen at run time with one variable; nothing else needs to be set:

| `KADATH_LINSOLVE` | Solver |
|---|---|
| `krylov` (default with MUMPS) | Newton-Krylov: GMRES on the matrix-free `J v`, preconditioned by MUMPS factors of an earlier Jacobian |
| `mumps` | exact sparse Newton step with MUMPS every iteration |
| `scalapack` (default without MUMPS) | dense LU (the original Kadath solver; memory grows as n²) |
