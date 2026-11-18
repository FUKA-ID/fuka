\mainpage FUKA Initial Data
# FUKAv2

Author(s)    : Samuel D. Tootle,  L. Jens Papenfort  
Maintainer(s): Samuel D. Tootle - tootle@itp.uni-frankfurt.de  
License      : GPLv3+ for all other code  

---

# 1. Overview

The FUKAv2 encompasses a rewrite of the original initial data (ID) solvers that were made publicly
available in 2021.  Although portions of the original codes have been reused, a significant
overhaul of the solvers has been done not only to make the interface more accessible to the user, 
but to also provide a coherent interface between the isolated and binary solvers.

Aside from the abstraction and interface, the key component that makes the v2 binary solvers
considerably more reliable and functional is the use of super-imposed boosted isolated objects
to construct the "initial guess" for the binary solver.  This allows *nearly* arbitrary configurations
to be constructed from scratch without iterative changes to the binary objects as was required in
the FUKAv1 binary solvers.

The *nearly* caveat is a function not only of resolution, but also the domain decomposition 
near the excision or stellar surfaces due to Gibbs phenomena that occur.  In the vast majority of test
cases this does not prohibit the construction of ID even at extreme mass ratios `q^{-1} > 20`.
It remains an on-going effort to optimize the domain decomposition to minimize these effects.

# 2. Nomenclature and Units

- In all FUKA solvers we work in geometric units such that `G = c = 1` unless otherwise specifically stated [^1]
- There will be mentions of *implicit solutions* throughout the various solver READMEs.  Each subsequent stage for
a given solver builds on the solution of previous stages.  These solutions usually have some useful characteristics 
(e.g. a non-rotating TOV solution prior to generating a rotating one; all stages of the isolated solutions when generating binary ID) 
which can be reused or analyzed later. However, these solutions do not match what has been requested by the user, 
they are simply a consequence of the ID solving process.
- In the case of binary ID, we use the convention throughout the documentation and within the FUKAv2 solvers that `q <= 1`
- When fixing the spin, it is common to use the dimensionless spin parameter `chi = S / M^2` where 
    - S is the spin angular momentum as measured (quasi)-locally on the compact object
    - `M` is the compact object's gravitational mass.  
        - In the case of NS ID, `M` is the ADM mass as computed from the isolated solution for a given rest mass and spin
        - In the case of BH ID, `M` is the Christodoulou mass
- For details on the formulations used within these solvers, please review the original [publication](https://arxiv.org/abs/2103.09911) 

[^1]: There exists a variable in the config file call `qpig` which is equivalent to `4*PI*G`.  This was originally designed to allow an arbitrary rescaling of the system of equations, matter, etc; however, this has not been implemented consistently as there was no clear benefit other than confusion when working in different unit systems

# 3. Organization

The solvers in `codes/FUKA`:

| Solver | Initial data |
|---|---|
| [BH](BH/README.md) | isolated (spinning, boosted) black hole |
| [NS](NS/README.md) | isolated non-rotating and uniformly rotating neutron star |
| [NS_DIFFROT](NS_DIFFROT/README.md) (beta) | isolated differentially rotating neutron star (XCTS) |
| [NS_isotropic](NS_isotropic/README.md) (beta) | rotating and differentially rotating neutron star in quasi-isotropic coordinates (2D) |
| [BBH](BBH/README.md) | binary black hole |
| [BNS](BNS/README.md) | binary neutron star |
| [BHNS](BHNS/README.md) | black hole - neutron star binary |
| `BNS_headon` (alpha) | head-on BNS (stars from rest, BNS solver's `headon` stage) |

The solvers are structured as follows:

1. Each directory contains the files needed for the respective solver
    1. `CMakeLists.txt` is the file needed by CMake to compile the codes
    2. `compile` is a symbolic link to the script stored in `$HOME_KADATH/build_release` to ease compiling
    3. `src` directory contains the relevant source files:
        - `solve.cpp`: the one and done solve code for each solver
        - `reader.cpp`: the reader can provide diagnostics from ID solutions that are computed from the ID
2. The related `solve.cpp` in most cases is simply a frontend with minimal functionality.  
Most of the business is stored in `$HOME_KADATH/include/FUKA_Solvers`: one directory per solver (`BH_XCTS`, `NS_XCTS`,
`BBH_XCTS`, `BNS_XCTS`, `BHNS_XCTS`) and the code they share in `utilities/`.  `$HOME_KADATH/include/Solvers` holds the
quasi-isotropic NS solver (`ns_isotropic`), shared sequence and system-of-equations tools, and the previous solver
generation, which is deprecated and will be removed
3. Each solver has its own `reader.cpp`

# 4. Getting Started

First and foremost, see \ref install for instructions on how to obtain and build
the FUKA library, initial data codes, and related PythonTools.

Once the intended features have been successfully compiled, the \ref tutorials are
a great place to get acquainted with FUKA!

# 5. Newton solvers, memory and number of nodes

Every solver stage is a Newton-Raphson iteration. Built with MUMPS (see \ref install), FUKA solves each
Newton step with a sparse Jacobian; nothing needs to be configured. `KADATH_LINSOLVE` selects the solver:

| `KADATH_LINSOLVE` | Solver | Use |
|---|---|---|
| `krylov` (default with MUMPS) | GMRES on the matrix-free Jacobian product, preconditioned by MUMPS factors of an earlier Jacobian | all runs |
| `mumps` | exact sparse Newton step every iteration | when Newton-Krylov struggles (it re-assembles the Jacobian every step, about twice as slow) |
| `scalapack` (default without MUMPS) | dense LU, the solver of FUKA <= v2.4 | small systems; memory grows as the square of the number of unknowns |

The three are alternatives for the same Newton iteration and give the same solution to the requested
`solver_precision`; only one is used per run (e.g. `export KADATH_LINSOLVE=mumps` before `mpirun`). Use the
default unless a run fails to converge. A Newton iteration that diverges (the residual becomes non-finite or grows by more
than a factor 1e6) stops with a message instead of trying the MUMPS fallback configurations, which cannot repair
it; check the initial data and configuration in that case.

The Newton solvers keep the Jacobian distributed for MUMPS: rank 0 gathers only a sample of large entries (about
10 % of a large matrix) to compute the column permutation that puts large entries on the diagonal, instead of the
whole matrix; later factorizations start from the previous permutation. `export KADATH_MUMPS_INPUT=centralized` gathers the whole matrix on rank 0 and lets MUMPS compute the
permutation (the previous behaviour; same solutions, more memory on rank 0). Rank 0 still runs the MUMPS analysis
of the whole sparsity pattern and needs the most memory of all ranks for the largest Jacobians: give it a node
with free memory.

If the MUMPS factors do not fit in memory (on some node the sum of its ranks' analysis estimates exceeds 80 % of
the node's free memory, or the factorization still runs out of memory at that limit), FUKA writes the factors to disk (MUMPS out-of-core)
and logs `MUMPS: OUT-OF-CORE: insufficient memory ...` with the directory used. The run continues but is slower,
since every solve, including each GMRES iteration, reads the factors back. `KADATH_MUMPS_OOC=off` makes such a run
stop with an insufficient-memory error instead, `KADATH_MUMPS_OOC=on` always uses the disk, and
`KADATH_MUMPS_OOC_DIR=<dir>` sets the directory (default `$TMPDIR`, else `/tmp`; a node-local disk is fastest).
On diskless nodes `/tmp` is held in memory, so the factors written there take the same RAM: check with
`df -T /tmp` in a job and point `KADATH_MUMPS_OOC_DIR` to a parallel file system instead.
Out-of-core lowers the factorization memory of every rank, not the analysis memory of rank 0.

## Memory

The memory of a run is the sum of two parts:

1. Every MPI rank holds the whole numerical space and system of equations. This part grows with the
   resolution and the number of domains and is the same for all three solvers. Measured: about 0.2 GB per
   rank for an isolated BH or NS at resolution 9, 0.35 GB at resolution 13, and 0.5 GB per rank for the
   default binaries.
2. The linear solver. Dense ScaLAPACK needs about 40 n^2 bytes in total (n unknowns: 46 GB for n = 34,815),
   independent of the number of ranks. Newton-Krylov and MUMPS store the sparse factors (0.1-0.7 GB for the
   cases below) and rank 0 additionally holds the sparsity pattern for the MUMPS analysis and the gathered
   sample of large entries, so the first node needs more memory than the others. Rank 0 peak, default
   (distributed) vs `KADATH_MUMPS_INPUT=centralized`: BBH res 11 6.6 vs 10.7 GB, BHNS res 11 11.9 vs 18.7 GB;
   the other ranks about 0.9-1.3 GB each. The table below was measured with centralized input.

The solvers print the peak memory per rank, per node and in total after every Newton step
(`Memory high-water mark (VmHWM)`), which is the quantity to size a run with.

Measured peak memory and wall time (192 GB, 40-core nodes; `outer_shells` adds domains around the binary):

| Case | n (largest stage) | ranks | Krylov: total / first node | MUMPS: total | ScaLAPACK: total |
|---|---|---|---|---|---|
| BH, res 9 | 10,471 | 40 | 7.0 GB | 7.1 GB | 10.8 GB |
| BH, res 11 | 20,249 | 40 | 9.1 GB | 9.8 GB | 23.9 GB |
| BH, res 13 | 34,815 | 40 | 14.0 GB | 14.5 GB | 59.9 GB |
| BBH, res 9 | 33,878 | 80 | 38 GB / 19 GB | 50 GB | 85 GB (res 7) |
| BBH, res 9, 2 outer shells | 39,332 | 80 | 44 GB / 24 GB | 54 GB | |
| BBH, res 9, 4 outer shells | 44,786 | 80 | 48 GB / 27 GB | 61 GB | |
| BBH, res 11 | 64,634 | 80 | 72 GB / 41 GB | 83 GB | (about 240 GB) |
| BHNS, res 11 (README second run) | 98,687 | 80 | 116 GB / 59 GB | | (about 500 GB) |
| BNS, res 11 (README second run) | 91,412 | 80 | 114 GB / 60 GB | | (about 450 GB) |

Values in parentheses are estimated from the 40 n^2 bytes of the dense solver plus the replicated part.

## Choosing the number of nodes

- With MUMPS the binary cases at resolution 9 to 11 run on 2 nodes (80 ranks); more ranks shorten the
  Jacobian assembly but add 0.2-0.5 GB per rank of replicated memory.
- Keep the first node's memory (`max ... GB per node` in the log) below the node memory; it is the largest.
  Rank 0 grows with the Jacobian's non-zeros: about 53 GB at 1.5e9 and 149 GB at 2.8e9 non-zeros (centralized
  input, 64-bit METIS for the latter; distributed input, the default now, lowers this by about a third). For such
  cases put rank 0 alone on the first node (e.g. a heterogeneous job, or a host file with one slot on the first
  node).
- Use `OMP_NUM_THREADS=1` (one MPI rank per core).

# 6. Acknowledgements

I have been developing FUKAv2 since before the initial public release in an 
effort to automate ID construction across the space of parameters.
To that end, I am very grateful to my original collaborators 
L. Jens Papenfort and Elias R. Most for their helpful discussions.  

Furthermore, I would like to thank Konrad Topolski for being an early adopter of the v2 codes and for his continued feedback and support throughout the many changes since
then.

# 7. Outstanding Tasks

The following are on the list of things to do based on expected level of effort:

1. Rewrite exporters to make them more readable and consistent
2. Add documentation for `PythonTools` to make them more accessible
3. Implement solvers for BH, BBH, and BHNS that allow for maximal spinning black holes

# 8. Contributing

1. Feedback and bugs are always a welcomed contribution.  One can always e-mail the maintainers for issues experienced or configurations that provide challenges as this can help to diagnose limitations in the solvers
2. Help with the current outstanding tasks is of course possible whether through code or experience with certain issues, e.g. optimizing a fix grid setup to avoid spectral artifacts.  For code contributions, forking and submitting pull requests is possible.