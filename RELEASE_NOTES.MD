# Frankfurt University/Kadath Initial Data (FUKA) - Release Notes

## v2.5.0
### New Features
* Sparse Newton solvers for all ID codes (requires MUMPS, see `installation.md`).
  The Jacobian of each Newton step is assembled as a sparse matrix instead of a dense ScaLAPACK
  matrix, so the solve no longer needs memory proportional to the square of the number of unknowns.
    * Default: Newton-Krylov. GMRES on the matrix-free Jacobian-vector product, preconditioned by
      MUMPS factors of an earlier Jacobian; the Jacobian is only re-assembled when the
      preconditioner no longer suffices (typically 1-2 assemblies per stage).
    * `KADATH_LINSOLVE` selects the solver at run time: `krylov` (default with MUMPS), `mumps`
      (exact sparse Newton step every iteration) or `scalapack` (the previous dense solver, the
      default when FUKA is built without MUMPS). No other settings are needed.
    * The default BBH and BNS CI cases run on 2 nodes instead of 10 and finish 3-6 times faster
      with the same solutions (see the PR / benchmark tables in the solver READMEs).
* Faster Jacobian assembly for every solver: `do_col_J` only re-evaluates definitions and
  operators that a column can change (unchanged subtrees are cached, linear operators act on the
  derivative only), and `do_JX` uses the same shortcuts.
* The peak memory (VmHWM per rank, per node and in total) is printed after every Newton step.
* Tests of the sparse solvers: `codes/tests/newton_sparse` (see its README).
### Bug Fixes
* Solutions, checkpoints and config files are written atomically (temporary file, `fsync`, rename) by
  `bco_utils::save_to_file` and `write_config`. Rank 0 rewrites e.g. `ns_regrid.dat` under the same name
  at every step and all ranks read it after a barrier; on a network file system ranks on other nodes could
  read the old or a partly written file and abort with `std::bad_alloc` (seen in `NS_DIFFROT` on 80 ranks).
* `Lorene_Table` stops with an error naming the file when no EOS entries are read, instead of corrupting
  the table sizes.
* BNS and BHNS always construct their neutron stars with the 3D XCTS solver: `use_iso_solver` from the
  binary config is no longer passed to the companion stars.
### Changes
* `NS_isotropic` reader: the GRV2 virial identity is no longer printed. Its integrand mixed the p/rho
  rescaled matter sources with unrescaled field terms, and the 1/(r sin(theta)) weight was applied by a
  spectral division that is not valid for terms that do not vanish on the axis, so the printed value
  was not the GRV2 residual (it stayed at O(1) for converged solutions at every resolution). A correct
  GRV2 needs a direct (r, theta) quadrature.
* `NS_isotropic` reader: the energy density used for the angular momentum volume integral now matches
  the solver (`E = W^2 p h - p delta` in the p/rho rescaled variables).
* Documentation: READMEs for `NS_DIFFROT` (differentially rotating stars, KEH law) and `NS_isotropic`;
  the example outputs of the solver READMEs were regenerated.

## v2.4.5
### Bug Fixes
* The top theta coefficient of axisymmetric (m = 0) `COS_EVEN` fields is no longer an
  unknown or a condition of `System_of_eqs` ([#14](https://github.com/FUKA-ID/fuka/issues/14)).
  Its theta derivative vanishes at every collocation point, so the mode is in the kernel of
  the discrete Laplacian and was set by aliased source terms. This affected rotating and
  high-spin ID.
    * Applies to the spherical (`Domain_nucleus`, `Domain_shell`, `Domain_compact`), adapted,
      polar and adapted polar domains, including the surface mapping of adapted domains.
      Other upstream Kadath domain families are unchanged.
    * Each `COS_EVEN` m = 0 field has one fewer theta mode per domain, so solutions differ
      from v2.4.4 at the level of the removed coefficient (round-off for non-rotating
      objects).
    * The mode is removed from the variables at the start of every Newton step, so
      solutions from earlier versions still load and resume.
    * The point-based matching `System_of_eqs::add_eq_matching_non_std` does not account
      for the removed mode on shells. FUKA does not use it.

## v2.4
### New Features
* Metadata related to FUKA version, commit, and build date is now part of
  the INFO file for each solution.

## v2.3
### Removed Features
* FUKAv1 Codes have been removed.
    * FUKA solvers are now stored in `codes/FUKA`
### New Features
* More robust algorithm for determining grid structure for BHNS and BBH systems
* Added support for GRHaYLEOS
    * Needs to be installed separately and the base root set as GRHAYL_ROOT environment variable
    * FUKA needs to be compiled with GRHAYL_EOS macro defined, e.g.
    `cmake -DPAR_VERSION=ON -DCMAKE_BUILD_TYPE=Release -DGRHAYL_EOS=ON  ..`
    * GRHaYLEOS support enables native support for stellar collapse 3D tables
* (beta) Multi-threaded support for **export only**
    * `include/memory.hpp`: Need to define `#define DEFAULT_KAD_MEM`
### Solvers in development
* (beta) Support for differentially rotating neutron star solutions in XCTS coordinates
* (beta) Support for differentially rotating neutron star solutions in Quasi-isotropic coordinates
* (alpha) Head-on initial data for BNS and BHNS
    * Black holes can be boosted

## v2.2
* With this release, user demanded shells around compact objects have been removed.
In its place is a significantly more robust solution
which determines an optimal configuration of spherical grids around each compact object in binary spaces.
* A minimal Config file has been introduced and is now the default for all solvers. In this way the user is only
concerned with the parameters necessary to generate initial data.
    * The "full" Config file must still be used for the ID solutions
    * The "full" Config file can be obtained by running `solve full` if desired for a new sequence
* (Beta) A 1D sequences are now supported such that a series of initial data solutions can be automated

## v2.1
This release includes a rewrite and reorganization of the Python readers for FUKA initial data among many bug fixes and quality of life changes.
The old python readers have thus been removed from the solvers and are now centrally located in the [codes/PythonTools](https://github.com/samueltootle/fuka/tree/fuka/codes/PythonTools) directory.

Along side of the FUKAv2.1 release is the official inclusion of the thorns to import FUKA initial data into the Einstein Toolkit - [ticket](https://bitbucket.org/einsteintoolkit/tickets/issues/2692/inclusion-of-fuka-importer-thorns). Please note the import thorns are not activated by default due to the need for Boost libraries.  For an example thorn to provide Boost libraries within the Einstein Toolkit, see [https://github.com/dradice/Boost](https://github.com/dradice/Boost).

## v2.0
The release of FUKAv2 is considerable step forward in reliable generation of extremal spinning, asymmetric
binary initial data using the KADATH spectral software.  The v2 solvers aim to maximize convergence by using super-imposed
isolated solutions to setup the initial guess for binary ID.  Additionally, v2 aims to automate the generation of ID by minimizing
the workflow for the user.  Finally, v2 allows quite for considerable flexibility in setting up the config file to make ID generation
as efficient as possible within the KADATH framework including reusing previously solved implicit isolated solutions.  For more details,
please see the documentation in the [FUKAv2](https://github.com/samueltootle/fuka/tree/fuka/codes/FUKAv2_Solvers/).

  - FUKAv1 Specific: There have been some core changes to various utilities used when constructing ID that have required some refactoring of the v1 solvers.  Spot testing has been done to ensure these codes function as originally intended - however - these codes, overall, remain unchanged.

## v1.0
The first release of the FUKA codes were released in 2021 alongside the code paper which can be found here https://arxiv.org/abs/2103.09911

The original codes can be found in [FUKAv1](https://github.com/samueltootle/fuka/tree/fuka/codes/FUKAv1_Solvers/) and exist primarily
for historical purposes and should not be used by new users.  However, those that wish to develop more complicated codes within
KADATH may find them useful as their structure is less abstract.

**NOTE: After the v2.0 release, these codes are no longer modified or maintained.**