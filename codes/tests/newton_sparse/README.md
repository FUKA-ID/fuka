# newton_sparse tests

Unit and integration tests of the sparse Newton solvers (`src_par/do_newton.cpp`, `src/Linear_solvers/`,
the Jacobian shortcuts in `src/System_of_eqs/solver.cpp` and `src/Ope_eq/ope_eq.cpp`).
Each test is a standalone executable that returns 0 on success, 1 on failure and 77 when
the library was built without MUMPS (`KADATH_USE_MUMPS` undefined).

| Test | Ranks | Checks |
|---|---|---|
| `test_sparse_triplets` | serial | block size rule, 1-D block-cyclic column ownership vs ScaLAPACK `numroc`/`indxl2g`, column to COO round trip (only exact zeros dropped), row/column maxima and `drop_negligible` |
| `test_gmres` | serial | right-preconditioned GMRES on dense systems: requested tolerance met by the true residual, one iteration with the exact preconditioner, few with a stale one, iteration cap, zero right-hand side |
| `test_mumps_wrapper` | 1, 2, 4 | `MumpsSolver` on a block-sparse matrix with bordering rows/columns and a zero diagonal: residual, error, identical solution on all ranks, instance reuse, centralized/distributed input, ParMETIS, independence of the column distribution, residual-check rejection, retry ladder, singular matrix detection |
| `test_newton_kadath` | 1, 4 | a nonlinear Kadath system with a `var_double` and an `eq_int` row (analytic solution `P = log(1 + a/r)`, `lam = 1/2`): `do_col_J` determinism (T3), `do_JX(e_j) == do_col_J(j)` incl. adapted and homothetic variable domains (T3b, T3c), the `do_col_J` / `do_JX` shortcuts against the full evaluation (T3d, T3e), one Newton update MUMPS vs ScaLAPACK (T4a), full solves with MUMPS and ScaLAPACK (T4b) and Newton-Krylov (T4d) against each other and the analytic solution |

## Build and run

```bash
# library first, built with MUMPS (see installation.md), then
cd $HOME_KADATH/codes/tests/newton_sparse && . compile
export OMP_NUM_THREADS=1
./bin/Release/test_sparse_triplets
./bin/Release/test_gmres
for np in 1 2 4; do mpirun -np $np ./bin/Release/test_mumps_wrapper; done
for np in 1 4; do mpirun -np $np ./bin/Release/test_newton_kadath; done
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
