\page nsdiffrot Differentially Rotating Neutron Star
# Differentially Rotating Neutron Star (NS_DIFFROT) Initial Data (beta)

# 1. Overview

The NS_DIFFROT code generates isolated neutron stars in the same conformally flat, maximally sliced XCTS
formulation as the NS solver, but the stars can be differentially rotating.  Differential rotation follows
the KEH (Komatsu, Eriguchi & Hachisu 1989, "j-constant") law

    Omega = Omega_c - j / A^2

where `j = u^t u_phi` is the specific angular momentum.  The angular velocity therefore decreases from its central
value `Omega_c` towards the surface over a length scale `A`: `A -> infinity` is uniform rotation, while `A` comparable
to the equatorial radius is strong differential rotation.  The hydrostatic equilibrium is given by the
corresponding first integral

    H + ln(N) - ln(W) - j^2 / (2 A^2) = const

with the log-enthalpy `H`, lapse `N` and Lorentz factor `W`.

Differentially rotating stars can be strongly deformed, therefore they are not fixed by the spin but by their shape:

- the mass (`madm` or `mb`)
- the axis ratio `R_ratio = R_p / R_e` (polar over equatorial coordinate radius of the surface)
- the degree of differential rotation `A_ratio = A / R_e`

The solver then determines `Omega_c`, `A`, `R_e` and the angular velocity field `Omega`.  The resulting spin,
angular momentum, masses and radii can be obtained from the reader.

# 2. Organization

1. `CMakeLists.txt` is the file needed by CMake to compile the codes
2. `compile` is a symbolic link to the script stored in `$HOME_KADATH/build_release` to ease compiling
3. `src` directory contains the relevant source files:
    - `solve.cpp`: the one and done solve code
    - `reader.cpp`: the reader can provide diagnostics from ID solutions that are computed from the ID
    - `setup_from_iso.cpp`: converts a solution of the quasi-isotropic solver ([NS_isotropic](../NS_isotropic/README.md)) into an XCTS initial guess

# 3. Base Usage

1. Generate the initial config file by running `solve` for the first time
2. Modify the initial config file based on ID characteristics you are interested in
3. Rerun (using parallelization) using this config file, e.g. `mpirun ./bin/Release/solve initial_ns.info`

# 4. Your first run!

1. Generate the initial config file by running `solve` for the first time
2. Rerun (using parallelization) using this config file, e.g. `mpirun ./bin/Release/solve initial_ns.info`
3. This will result in the generation of a pair of files containing the solution: `NS_DIFF_ROT.togashi.keh.madm.1.4.Ar.1e+06.Rr.1.0.09.<info/dat>`

We can deconstruct the name to make it understandable:

- `NS_DIFF_ROT.` denotes a converged NS solution after the `DIFF_ROT` stage
- `togashi` denotes the eosfile used in the ID construction
- `keh` denotes the differential rotation law
- `madm.1.4.`: the solution is fixed by an ADM mass of 1.4M
- `Ar.1e+06.Rr.1.`: `A_ratio = 1e6` and `R_ratio = 1`
- `0.09`: the number of *additional* spherical `nshells` is `0` and the resolution in each domain is `9` collocation points in the radial and theta direction with `8` points in the phi direction
- `info`: The info file contains all the steering parameters, values used in creating the domain decomposition, EOS parameters, stored fields, stages, settings, and controls
- `dat`: The dat file is a binary file that contains the numerical space and the variable fields

The default configuration has `A_ratio = 1e6` and `R_ratio = 1`, i.e. a star rotating uniformly with a
spherical surface, which is a (nearly) non-rotating star (`chi = 0.0003`).  Aside from the diagnostics observed during the solver
stage, we can use the reader to verify the ID.  This can be done by running:

`./bin/Release/reader NS_DIFF_ROT.togashi.keh.madm.1.4.Ar.1e+06.Rr.1.0.09.info`

Which results in the following:

```
                   RES = [+9,+9,+8]
            Coord R_IN = +3.16993
               Coord R = [+6.33299, +6.33299]
           Coord R_OUT = +10.25809

               Areal R = +7.81037 [+11.53591km]
     Circumferential R = +7.81037 [+11.53591km]
         Mass Shedding = +1.00000
         Baryonic Mass = +1.55246
              ADM Mass = +1.40000 [+1.40000]
          ADM Momentum = +0.00054
                   Chi = +0.00028 [+0.00028]
                 Omega = +0.00002
       Central Density = +1.37465e-03
        Central log(h) = +2.31926e-01
      Central Pressure = +2.34625e-04
    Central dlog(h)/dx = +3.08707e-12
Central Euler Constant = -2.22178e-01
     Integrated log(h) = +186.26441
                  Beta = +0.00000

                    Mk = +1.40117, Diff: +8.32762e-04
                    Px = +0.00000
                    Py = +0.00000
                    Pz = +0.00000
```

<b>
Note: brackets denote the values stored in the config file that the solution was fixed by.
</b>

# 5. Understanding the NS_DIFFROT INFO file

Using your favorite text editor, you can open up the `initial_ns.info`.  We will go through the file,
but we'll discuss only the details relevant to differential rotation.  The EOS parameters, fields and sequence
settings are the same as for the [NS solver](../NS/README.md).

## NS Fixing parameters

```
ns
{
    chi 0
    madm 1.3999999999999999
    res 9
    eosfile togashi.lorene
    eostype Cold_Table
    differential_rotation
    {
        A_ratio 1000000
        R_ratio 1
        law keh
    }
}
```

- `madm`: the ADM mass of the star (`mb` can be used to fix the baryonic mass instead)
- `chi`: the dimensionless spin, only used by the `uniform_rotation` stage
- `res`: the final global resolution of interest
- `differential_rotation`:
    - `law`: the rotation law, currently `keh`
    - `R_ratio`: `R_p / R_e`, the axis ratio of the surface (`< 1`)
    - `A_ratio`: `A / R_e`; `1e6` is uniform rotation, `~1` is strong differential rotation

## Stages

```
stages
{
    differential_rotation on
    norot_bc on
    uniform_rotation on
}
```

- `norot_bc`: the non-rotating (TOV) solution
- `uniform_rotation`: a uniformly rotating solution fixed by `chi`
- `differential_rotation`: a differentially rotating solution fixed by `R_ratio` and `A_ratio`
- `total_bc`: accepted for configs from earlier versions and read as `uniform_rotation`

The last enabled stage is the one that is solved for.

## Relevant Controls

```
sequence_controls
{
    checkpoint off
    sequences on
    use_iso_solver on
}
```

- `checkpoint`: this will result in checkpoints being saved to file during each solving iteration
- `sequences`: this toggle is enabled by default and essentially tells the driver routine to start from scratch
- `use_iso_solver`: construct the initial guess with the 2D quasi-isotropic solver (recommended, see below)

# 6. Your Second run!

Now that you've generated the simplest case and we have a better understanding of the config file, we can try
something more interesting: a strongly differentially rotating star.

1. Open the initial config file in your favorite editor
2. Set
    - `madm 1.6`
    - `res 13`
    - `A_ratio 1`
    - `R_ratio 0.6`
3. Run (using parallelization) using this config file, e.g. `mpirun -np 40 ./bin/Release/solve initial_ns.info`

The steps of the quasi-isotropic solver (`R_ratio = 0.9 ... 0.6`), the conversion to XCTS and the XCTS solutions at
the resolutions 9, 11 and 13 take about 5 minutes on one 40-core node.  This results in the converged dataset of
`NS_DIFF_ROT.togashi.keh.madm.1.6.Ar.1.Rr.0.6.0.13.info/dat`.

We can of course verify that the ID matches our expectation using

`./bin/Release/reader NS_DIFF_ROT.togashi.keh.madm.1.6.Ar.1.Rr.0.6.0.13.info`

```
                   RES = [+13,+13,+12]
            Coord R_IN = +2.26125
               Coord R = [+4.52691, +7.54485]
           Coord R_OUT = +12.19594

               Areal R = +8.57376 [+12.66344km]
     Circumferential R = +9.30371 [+13.74157km]
         Mass Shedding = +0.54481
         Baryonic Mass = +1.75670
              ADM Mass = +1.60000 [+1.60000]
          ADM Momentum = +1.82151
                   Chi = +0.71153 [+0.71153]
                 Omega = +0.06535
       Central Density = +1.16317e-03
        Central log(h) = +1.67141e-01
      Central Pressure = +1.34110e-04
    Central dlog(h)/dx = +1.44034e-14
Central Euler Constant = -2.95147e-01
     Integrated log(h) = +193.95601
                  Beta = +0.11879

                    Mk = +1.59972, Diff: +1.74204e-04
                    Px = -0.00000
                    Py = -0.00000
                    Pz = -0.00000
```

`Coord R` is the [polar, equatorial] coordinate radius of the surface (`4.527 / 7.545 = 0.6`), and the resulting
spin is `chi = 0.71`.  `Mass Shedding` is the ratio of the radial log-enthalpy gradients at the equator and the
pole: 1 for a spherical star and 0 at the mass-shedding limit, where the equatorial surface forms a cusp.

# 7. How NS_DIFFROT ID is Generated

## Initial Setup with the quasi-isotropic solver

With `use_iso_solver on` (the default), the star is first solved with the 2D (axisymmetric) quasi-isotropic
solver of [NS_isotropic](../NS_isotropic/README.md):

1. the non-rotating (TOV) solution
2. a slowly, uniformly rotating solution (`chi = 0.1`)
3. the differentially rotating solution

These solutions are inexpensive (2D) and are kept as `NS_ISO_*` files.

### Automated iterative axis ratio

Large changes of the shape do not converge.  Therefore, if `R_ratio < 0.9`, the axis ratio starts at `0.9` and
is decreased in steps of `0.05` to the requested value.  Each step is solved, the grid is re-adapted to the new
surface, and the step is solved again, therefore there are two `NS_ISO_DIFF_ROT` files per axis ratio.  The number
before the resolution in their names is `Omega_c`.

## DIFF_ROT Stage

The final quasi-isotropic solution is interpolated onto the XCTS variables (conformal factor, lapse, shift,
log-enthalpy and `Omega`) of an adapted 3D grid, and the `DIFF_ROT` stage is solved.

### Automated resolution increase

Once the above procedures are completed and the input `res` is higher than the `initial_resolution`, the
solution is interpolated onto the higher resolution numerical grid(s) and the stage is solved again.

## Without the quasi-isotropic solver

With `use_iso_solver off`, the XCTS solver first computes a TOV and a slowly rotating (`chi = 0.1`) solution and
then solves the `DIFF_ROT` stage directly at the requested `R_ratio`.  There is no automated iterative axis ratio
in 3D, so this only converges for mildly deformed stars.

Existing quasi-isotropic solutions can also be converted by hand using

    ./bin/Release/setup_from_iso NS_ISO_DIFF_ROT.<...>.info

which writes `initns_xcts.info/dat`, an XCTS initial guess that `solve` accepts.

## Uniformly rotating stars

Disabling `differential_rotation` and enabling `uniform_rotation` computes uniformly rotating stars fixed by `chi`
in the same way as the [NS solver](../NS/README.md), including the automated iterative spin and mass.

<b>
Notes:

1. The Newton steps use the sparse solvers described in the [FUKA README](../README.md) (section 5); run with
`OMP_NUM_THREADS=1`.
</b>
