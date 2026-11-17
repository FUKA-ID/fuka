/*
 * Copyright 2022
 * This file is part of the KADATH library and published under
 * https://arxiv.org/abs/2103.09911
 *
 * Author:
 * Samuel D. Tootle <tootle@itp.uni-frankfurt.de>
 * L. Jens Papenfort <papenfort@th.physik.uni-frankfurt.de>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */
#include "EOS/FUKA_EOS_Utilities.hh"
#include "bco_utilities.hpp"
#include "mpi.h"

/**
 * \addtogroup BNS_XCTS
 * \ingroup FUKA
 * @{*/

namespace Kadath {
namespace FUKA_Solvers {
namespace bco_u = ::Kadath::bco_utils;

template <class eos_t>
void BNS_XCTS<eos_t>::setup_hydro_rescaling_syst() {
    if (rank == 0) {
        if (solver_stage == STAGES::ECC_RED) {
            std::cout << std::string(42, '#') << std::endl
                      << "Eccentricity reduction step with fixed omega, chi = "
                      << (*bconfig)(CHI, BCO1) << "," << (*bconfig)(CHI, BCO2)
                      << " and q = " << (*bconfig)(Q) << std::endl
                      << std::string(42, '#') << std::endl;
        } else if (solver_stage == STAGES::PRE) {
            std::cout << std::string(42, '#') << std::endl
                      << "Hydro Rescaling stage \n"
                      << "using fixed orbital velocity with \n"
                      << "chi = " << (*bconfig)(CHI, BCO1) << ","
                      << (*bconfig)(CHI, BCO2) << " and q = " << (*bconfig)(Q)
                      << std::endl
                      << std::string(42, '#') << std::endl;
        } else if (solver_stage == STAGES::HEADON) {
            std::cout << std::string(42, '#') << std::endl
                      << "Head-on collision of two stars from rest \n"
                      << "with chi = " << (*bconfig)(CHI, BCO1) << ","
                      << (*bconfig)(CHI, BCO2) << " and q = " << (*bconfig)(Q)
                      << std::endl
                      << std::string(42, '#') << std::endl;
        }
    }
    if (solver_stage == STAGES::ECC_RED)
        binary_syst::set_ecc_red_orbit(*bconfig,
                                       rank,
                                       (*bconfig)(BCO_PARAMS::MADM,
                                                  NODES::BCO1),
                                       (*bconfig)(BCO_PARAMS::MADM,
                                                  NODES::BCO2));

    // setup background position vector field - only needed for ECC_RED stage
    Vector CART(*space, CON, *basis);
    CART = cfields->cart();

    // residual scaling factors for the enthalpy
    // used to correct the baryonic mass
    H_scale1 = 0;
    H_scale2 = 0;

    // set the shift of the "center of mass" along the y-axis
    // to zero, potentially fixing x-components of the ADM linear
    // momentum at infinity introduced by the eccentricity reduction
    // parameters
    if (std::isnan(bconfig->set(BIN_PARAMS::COMY)))
        bconfig->set(BIN_PARAMS::COMY) = 0.;

    // "background", unscaled logarithmic enthalpy from the previous step
    logh_const.reset(new Scalar(*logh));
    logh_const->std_base();

    Kadath::update_fields(*cfields, *coord_vectors, {}, xo, xc1, xc2);

    // setup a system of equations
    syst.reset(new System_of_eqs(*space, 0, ndom - 1));

    // constant "background" enthalpy
    syst->add_cst("Hconst", *logh_const);

    // residual scaling factors for the ethalpy
    syst->add_var("Hscale1", H_scale1);
    syst->add_var("Hscale2", H_scale2);

    // definition of H
    // - rescaled in the star
    // - constant everywhere else
    for (int d = 0; d < ndom; ++d) {
        if ((d >= space->ADAPTED2 + 1) ||
            ((d >= space->ADAPTED1 + 1) && (d < space->NS2)))
            syst->add_def(d, "H  = Hconst");
    }
    // inside the stars, it is the constant part
    // and a small correction factor to fix the correct
    // baryonic mass, due to slightly changing
    // fluid velocities
    for (int d = space->NS1; d <= space->ADAPTED1; ++d) {
        syst->add_def(d, "H  = Hconst * (1. + Hscale1)");
    }
    for (int d = space->NS2; d <= space->ADAPTED2; ++d) {
        syst->add_def(d, "H  = Hconst * (1. + Hscale2)");
    }

    Kadath::FUKA_EOS::set_eos_ope_struct<eos_t>()(*syst, eos_param);

    // populate all the boiler-plate constants, variables, and definitions
    syst_init();

    bool fixed_com = this->solver_stage == STAGES::HEADON;
    if (fixed_com) {
        // "center of mass" on the x-axis, connecting both stellar centers
        // These are fixed on the initial ID import as integrals at INF
        // cause instabilities in the initial solution
        syst->add_cst("xaxis", (*bconfig)(BIN_PARAMS::COM));
        syst->add_cst("yaxis", (*bconfig)(BIN_PARAMS::COMY));
    } else {
        // "center of mass" on the x-axis, connecting both stellar centers
        // fixed by the vanishing of the ADM linear momentum at infinity
        syst->add_var("xaxis", (*bconfig)(BIN_PARAMS::COM));

        // same on the y-axis in case finite momenta develope by
        // the eccentricity reduction parameters
        syst->add_var("yaxis", (*bconfig)(BIN_PARAMS::COMY));
    }

    // no additional force-balance is computed,
    // the matter distribution is fixed modulo the scaling factos above,
    // therefore the orbital frequency is a constant similar to
    // eccentricity reduced ID
    syst->add_cst("ome", (*bconfig)(BIN_PARAMS::GOMEGA));

    // orbital rotation corrected by the "center of mass" shifts, full shift
    // including the inspiral (eccentricity reduction)
    if (solver_stage == STAGES::ECC_RED)
        binary_syst::add_orbital_shift(*syst,
                                       true,
                                       &bconfig->set(BIN_PARAMS::ADOT),
                                       &CART);
    else
        binary_syst::add_orbital_shift(*syst, true);

    bool const corot = bconfig->control(CONTROLS::COROT_BIN);
    for (int d = 0; d < ndom; d++) {
        if ((d >= space->ADAPTED2 + 1) ||
            ((d >= space->ADAPTED1 + 1) && (d < space->NS2)))
            binary_syst::add_vacuum_eqdefs(*syst, d, corot, false);
        else
            binary_syst::add_matter_eqdefs(*syst, d, corot);
    }
    binary_syst::add_metric_equations(*syst, *space, ndom);

    // the stellar surfaces are where the matter vanishes
    syst->add_eq_bc(space->ADAPTED1, OUTER_BC, "H = 0");
    syst->add_eq_bc(space->ADAPTED2, OUTER_BC, "H = 0");

    if (!corot) {
        syst->add_eq_bc(space->ADAPTED1, OUTER_BC, "V^i * D_i H = 0");
        syst->add_eq_bc(space->ADAPTED2, OUTER_BC, "V^i * D_i H = 0");
        binary_syst::add_velocity_potential_eqs(*syst,
                                                space->NS1,
                                                space->ADAPTED1);
        binary_syst::add_velocity_potential_eqs(*syst,
                                                space->NS2,
                                                space->ADAPTED2);

        if (std::isnan(bconfig->set(BCO_PARAMS::FIXED_BCOMEGA, NODES::BCO1)))
            space->add_eq_int_outer_sphere_one(
                *syst,
                "integ(intS1) / Madm1 / Madm1 = chi1");

        if (std::isnan(bconfig->set(BCO_PARAMS::FIXED_BCOMEGA, NODES::BCO2)))
            space->add_eq_int_outer_sphere_two(
                *syst,
                "integ(intS2) - chi2 * Madm2 * Madm2 = 0 ");
    }

    space->add_eq_int_volume(*syst,
                             space->NS1,
                             space->ADAPTED1,
                             "integvolume(intMb) = Mb1");

    space->add_eq_int_volume(*syst,
                             space->NS2,
                             space->ADAPTED2,
                             "integvolume(intMb) = Mb2");

    // stage == ECC_RED
    if (!fixed_com) {
        space->add_eq_int_inf(*syst, "integ(intPx) = 0");
        space->add_eq_int_inf(*syst, "integ(intPy) = 0");
    }
}

/** @}*/
}  // namespace FUKA_Solvers
}  // namespace Kadath