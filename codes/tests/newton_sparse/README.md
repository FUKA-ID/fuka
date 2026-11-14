# newton_sparse tests

Quick checks for the sparse (MUMPS) Newton-Raphson path in `src_par/do_newton.cpp`.
Each test is a standalone executable that returns 0 on success, 1 on failure and 77 when
the library was built without MUMPS (`KADATH_USE_MUMPS` undefined).

| Test | Ranks | Checks |
|---|---|---|
| `test_sparse_triplets` | serial | block size rule, 1-D block-cyclic column ownership vs ScaLAPACK `numroc`/`indxl2g`, column to COO round trip (only exact zeros dropped) |
| `test_mumps_wrapper` | 1, 2, 4 | `MumpsSolver` on a block-sparse matrix with bordering rows/columns and a zero diagonal: residual, error, identical solution on all ranks, instance reuse, centralized/distributed input, ParMETIS, roundoff-level independence of the column distribution, singular matrix detection |
| `test_newton_kadath` | 1, 4 | a nonlinear Kadath system with a `var_double` and an `eq_int` row (analytic solution `P = log(1 + a/r)`, `lam = 1/2`): `do_col_J` determinism, one Newton update MUMPS vs ScaLAPACK, full solves with both backends, ScaLAPACK fallback when MUMPS fails (bitwise identical to the ScaLAPACK solve) |

## Build and run

```bash
# library first (with MUMPS_ROOT set, see installation.md), then
cd $HOME_KADATH/codes/tests/newton_sparse && . compile
export OMP_NUM_THREADS=1
./bin/Release/test_sparse_triplets
mpirun -np 4 ./bin/Release/test_mumps_wrapper
mpirun -np 4 ./bin/Release/test_newton_kadath
```

All three finish in well under a minute.

## Run-time switches of the Newton solver

| Variable | Values | Effect |
|---|---|---|
| `KADATH_LINSOLVE` | `mumps` (default with MUMPS), `scalapack` | Newton-Raphson linear solver |
| `KADATH_MUMPS_FALLBACK` | `scalapack` (default), `abort` | what a failed MUMPS step does: redo it with ScaLAPACK (warning printed) or abort |
| `KADATH_MUMPS_MAX_MEM_MB` | MB, default 0 (MUMPS estimate) | ICNTL(23) per-rank working-memory budget |
| `KADATH_NEWTON_PROFILE` | `1` | print do_col_J / do_JX / values-only residual timings and the J dx consistency each step |
| `KADATH_NEWTON_DUMP_PATTERN` | path prefix | write the Jacobian of the first `KADATH_NEWTON_DUMP_STEPS` (3) steps with ≥ `KADATH_NEWTON_DUMP_MIN_N` unknowns; analyse with `replay_jacobian` |
| `KADATH_LINSOLVE_COMPARE` | `1` | also solve each step with ScaLAPACK and print the relative difference (needs dense memory) |
| `KADATH_MUMPS_INPUT` | `centralized` (default), `distributed` | gather the matrix on rank 0 (allows the zero-free-diagonal permutation ICNTL(6)) or pass it distributed (ICNTL(18)=3) |
| `KADATH_MUMPS_ORDERING` | `metis` (default), `parmetis`, `amd`, `amf`, `pord`, `qamd`, `auto` | fill-reducing ordering |
| `KADATH_MUMPS_MEM_RELAX` | percent, default 30 | initial ICNTL(14); doubled automatically on workspace errors |
| `KADATH_MUMPS_VERBOSITY` | 0-4, default 1 | ICNTL(4) |
| `KADATH_MUMPS_OMP_THREADS` | default 1 | ICNTL(16) |
| `KADATH_MUMPS_BLR` | tolerance, default 0 (off) | block low-rank factorization (lossy) |
| `KADATH_MUMPS_OOC` | `1` | out-of-core factorization (`MUMPS_OOC_TMPDIR`) |
| `KADATH_MUMPS_COLUMN_BLOCK` | default 1 | block size of the column distribution used for assembly (1 = round robin, balances the per-domain cost of `do_col_J`) |
| `KADATH_MUMPS_PERMUTATION` | default 7 | ICNTL(6) column permutation (centralized input) |
| `KADATH_MUMPS_SCALING` | default: 0 centralized, 77 distributed | ICNTL(8); centralized + 77 (matching-based scaling) gave silently wrong solutions on BH Jacobians |
| `KADATH_MUMPS_RESIDUAL_TOL` | default 1e-10 | reject a MUMPS solution with larger max\|Ax-b\|/max\|b\| (then fallback/abort) |
| `KADATH_MUMPS_PIVOT_THRESHOLD` | default 0.1 | CNTL(1); the MUMPS default 0.01 can lose accuracy through pivot growth |
