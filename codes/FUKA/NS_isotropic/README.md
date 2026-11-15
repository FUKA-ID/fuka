\page nsisotropic Rotating Neutron Star in Quasi-Isotropic Coordinates
# Rotating Neutron Star in Quasi-Isotropic Coordinates (NS_isotropic) Initial Data (beta)

# 1. Overview

The NS_isotropic code generates stationary, axisymmetric neutron stars (non-rotating, uniformly rotating and
differentially rotating) in quasi-isotropic coordinates

    ds^2 = -N^2 dt^2 + A^2 (dr^2 + r^2 dtheta^2) + B^2 r^2 sin^2(theta) (dphi - omega dt)^2

on a 2D `(r, theta)` numerical grid with an adapted stellar surface.  This is the classic formulation of rotating
relativistic stars (no conformal flatness): it is exact for stationary axisymmetric stars, inexpensive (2D), and
robust for strongly deformed and differentially rotating configurations.  Its solutions are used by the
[NS_DIFFROT](../NS_DIFFROT/README.md) solver as initial guesses for the 3D XCTS stars (`use_iso_solver on`),
and can be used on their own, e.g. to explore sequences of rotating stars.

Differential rotation follows the KEH (j-constant) law `Omega = Omega_c - j / A^2`; see the
[NS_DIFFROT README](../NS_DIFFROT/README.md) for the law and its parameters.

# 2. Organization

1. `CMakeLists.txt` is the file needed by CMake to compile the codes
2. `compile` is a symbolic link to the script stored in `$HOME_KADATH/build_release` to ease compiling
3. `src` directory contains the relevant source files:
    - `solve.cpp`: the one and done solve code
    - `reader.cpp`: the reader can provide diagnostics from ID solutions that are computed from the ID

# 3. Base Usage

1. Generate the initial config file by running `solve` for the first time
2. Modify the initial config file based on ID characteristics you are interested in
3. Rerun (using parallelization) using this config file, e.g. `mpirun ./bin/Release/solve initial_2dns.info`

Since the problem is 2D, a few tens of MPI ranks (e.g. `mpirun -np 20`) are enough.

# 4. Your first run!

1. Generate the initial config file by running `solve` for the first time
2. Rerun (using parallelization) using this config file, e.g. `mpirun -np 20 ./bin/Release/solve initial_2dns.info`
3. This will result in the generation of a pair of files containing the solution:
`NS_ISO_UNIFORM_ROT.gam2.madm.1.4.chi.0.0.09.<info/dat>`

We can deconstruct the name to make it understandable:

- `NS_ISO_UNIFORM_ROT.` denotes a converged quasi-isotropic NS solution after the `uniform_rotation` stage
- `gam2` denotes the eosfile used in the ID construction (a `Gamma = 2` polytrope)
- `madm.1.4.chi.0.`: In this case, we have a NS with an ADM mass of 1.4M and a dimensionless spin of zero
- `0.09.`: the number of *additional* spherical `nshells` is `0` and the resolution in each domain is `9` collocation points in the radial and theta direction
- `info`: The info file contains all the steering parameters, values used in creating the domain decomposition,
EOS parameters, stored fields, stages, settings, and controls
- `dat`: The dat file is a binary file that contains the numerical space and the variable fields

The default configuration is a non-spinning star with an ADM mass of 1.4M.
Aside from the diagnostics observed during the solver stage, we can use the reader
to verify the ID.  This can be done by running:

`./bin/Release/reader NS_ISO_UNIFORM_ROT.gam2.madm.1.4.chi.0.0.09.info`

Which results in the following:

```
                   RES = [+9,+9]
            Coord R_IN = +4.06298
               Coord R = [+8.12595, +8.12595]
           Coord R_OUT = +13.14807

     Circumferential R = +9.58625 [+14.15889km]
         Mass Shedding = +1.00000
         Baryonic Mass = +1.50599
              ADM Mass = +1.40000 [+1.40000, +1.40001]
          ADM Momentum = +0.00000 [+0.00000]
                   Chi = +0.00000 [+0.00000]
                 Omega = +0.00000
       Central Density = +1.27959e-03
             Central h = +1.25592e+00
        Central log(h) = +2.27866e-01
      Central Pressure = +1.63734e-04
                   Beta: +0.00000

                    Mk = +1.40000, Diff: +5.33145e-10 [+1.56212e-06, +9.99096e-06]
```

<b>
Notes:

1. `Coord R` is the [polar, equatorial] coordinate radius of the surface.
2. `ADM Mass` is the surface integral of `B` at infinity; the brackets give the surface integral of `A` and the
full ADM surface integral.  `ADM Momentum` is the angular momentum as a surface integral [as a volume integral].
3. `Mk` is the Komar mass; `Diff` is its relative difference to the `ADM Mass` [to the two other ADM integrals].
4. `Chi` and `Omega` are the solution's spin and (central) angular velocity; the bracket after `Chi` is the value
stored in the config file.
</b>

# 5. Understanding the NS_isotropic INFO file

Using your favorite text editor, you can open up the `initial_2dns.info`.  We will go through the file,
but we'll discuss only the details relevant to the quasi-isotropic star.  The EOS parameters, controls and
sequence settings are the same as for the [NS solver](../NS/README.md).

## NS Fixing parameters

```
ns
{
    chi 0
    madm 1.3999999999999999
    res 9
    eosfile gam2.polytrope
    eostype Cold_PWPoly
    differential_rotation
    {
        A_ratio 1000000
        R_ratio 1
        law keh
    }
}
```

- `chi`: the dimensionless spin, used by the `uniform_rotation` stage
- `madm`: the ADM mass of the star (`mb` can be used to fix the baryonic mass instead)
- `res`: the final resolution used in each numerical domain
- `differential_rotation`: only used by the `differential_rotation` stage
    - `law`: the rotation law, currently `keh`
    - `R_ratio`: `R_p / R_e`, the axis ratio of the surface (`< 1`)
    - `A_ratio`: `A / R_e`; `1e6` is uniform rotation, `~1` is strong differential rotation

## Stages

```
stages
{
    differential_rotation off
    norot_bc on
    uniform_rotation on
}
```

- `norot_bc`: the non-rotating (TOV) solution, first with a fixed radius (`NS_ISO_NOROT_FIXED`), then with a
resolved surface (`NS_ISO_NOROT_BC`)
- `uniform_rotation`: a uniformly rotating solution fixed by `chi` (`NS_ISO_UNIFORM_ROT`)
- `differential_rotation`: a differentially rotating solution fixed by `R_ratio` and `A_ratio` (`NS_ISO_DIFF_ROT`)

The last enabled stage is the one that is solved for.

## Sequence Settings

```
sequence_settings
{
    solver_max_iterations 20
    solver_precision 1e-08
}
```

- `solver_max_iterations`: set the number of iterations not to exceed
- `solver_precision`: Determines what is the maximum precision allowed by the solver that determines whether or not the solution has converged

# 6. Your Second run!

Now that you've generated the simplest case and we have a better understanding of the config file, we can try
something more interesting: a strongly differentially rotating star.

1. Open the initial config file in your favorite editor
2. Set
    - `chi 0.1`
    - `res 13`
    - `A_ratio 1`
    - `R_ratio 0.6`
    - `differential_rotation on`
3. Run (using parallelization) using this config file, e.g. `mpirun -np 20 ./bin/Release/solve initial_2dns.info`

Many things will run differently this time!  Most notably are the following:

1. The TOV solution and a slowly, uniformly rotating solution (`chi 0.1`) are computed first
2. The axis ratio starts at `0.9` and is decreased in steps of `0.05` to `0.6` (see below)
3. Once the desired `R_ratio` is achieved, the solution is interpolated onto the numerical grids with the
resolutions 11 and 13 and the stage is solved again

This takes about 20 seconds on 20 MPI ranks and results in the converged dataset of
`NS_ISO_DIFF_ROT.gam2.keh.madm.1.4.Ar.1.Rr.0.6.0.0329427.0.13.info/dat`.  The number before the resolution
(`0.0329427`) is the central angular velocity `Omega_c` of the solution.

We can of course verify that the ID matches our expectation using

`./bin/Release/reader NS_ISO_DIFF_ROT.gam2.keh.madm.1.4.Ar.1.Rr.0.6.0.0329427.0.13.info`

```
                   RES = [+13,+13]
            Coord R_IN = +3.59268
               Coord R = [+7.18519, +11.97532]
           Coord R_OUT = +19.37693

     Circumferential R = +13.48374 [+19.91548km]
         Mass Shedding = +0.47074
         Baryonic Mass = +1.47896
              ADM Mass = +1.40000 [+1.39299, +1.42471]
          ADM Momentum = +1.60146 [+1.60146]
                   Chi = +0.81707 [+0.81707]
                 Omega = +0.03294
       Central Density = +6.01177e-04
             Central h = +1.12024e+00
        Central log(h) = +1.13539e-01
      Central Pressure = +3.61414e-05
                   Beta: +0.11672

                    Mk = +1.40000, Diff: +6.46657e-12 [+5.01945e-03, +1.74982e-02]
```

`Coord R` gives `7.185 / 11.975 = 0.6`, and the resulting spin is `chi = 0.82`.  The ADM mass from `B` and the
Komar mass agree to `1e-11`; the integrals of `A` and the full ADM integral are less accurate for strongly
deformed stars.  `Mass Shedding` is the ratio of the radial log-enthalpy gradients at the equator and the pole:
1 for a spherical star and 0 at the mass-shedding limit.

# 7. How NS_isotropic ID is Generated

## NOROT_BC Stage

The initial guess is a TOV solution for the input mass.  It is solved first with a fixed stellar radius
(`NS_ISO_NOROT_FIXED`) and then with the surface adapted to the star (`NS_ISO_NOROT_BC`).

## UNIFORM_ROT Stage

A uniformly rotating solution is found based on the `chi` fixing parameter.

### Automated iterative spin

As for the [NS solver](../NS/README.md), a spin `chi = +/- 0.1` is solved for before spins above `0.2`.
Spins above `0.7` are not supported.

### Automated iterative mass

In the event the desired mass is above the maximum TOV mass, the rotating solution is first constructed at the
maximum TOV mass and the mass is then increased to the desired value.

## DIFF_ROT Stage

### Automated iterative axis ratio

Large changes of the shape do not converge.  Therefore, if `R_ratio < 0.9`, the axis ratio starts at `0.9` and
is decreased in steps of `0.05` to the requested value.  Each step is solved, the grid is re-adapted to the new
surface, and the step is solved again, therefore there are two `NS_ISO_DIFF_ROT` files per axis ratio.

## Automated resolution increase

Once the above procedures are completed for the last stage and the input `res` is higher than the
`initial_resolution`, the solution is interpolated onto the higher resolution numerical grid(s) and the last stage
is solved again.

<b>
Notes:

1. The Newton steps use the sparse solvers described in the [FUKA README](../README.md) (section 5); run with
`OMP_NUM_THREADS=1`.
2. The XCTS initial guess of [NS_DIFFROT](../NS_DIFFROT/README.md) is built from these solutions with its
`setup_from_iso` code.
</b>
