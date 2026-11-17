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
void BNS_XCTS<eos_t>::setup_hydrostatic_equilibrium_stage() {
    if (rank == 0) {
        std::cout << std::string(42, '#') << std::endl
                  << "TOTAL - Hydrostatic equilibrium stage\n"
                  << "with mass fixing, chi = "
                  << (*bconfig)(BCO_PARAMS::CHI, NODES::BCO1) << ","
                  << (*bconfig)(BCO_PARAMS::CHI, NODES::BCO2)
                  << " and q = " << (*bconfig)(Q) << std::endl
                  << std::string(42, '#') << std::endl;
    }

    {
        double loghc = bco_u::get_boundary_val(space->NS1, *logh, INNER_BC);
        bconfig->set(BCO_PARAMS::HC, NODES::BCO1) = std::exp(loghc);
    }
    {
        double loghc = bco_u::get_boundary_val(space->NS2, *logh, INNER_BC);
        bconfig->set(BCO_PARAMS::HC, NODES::BCO2) = std::exp(loghc);
    }

    // setup background position vector field - only needed for ECC_RED stage
    Kadath::update_fields(*cfields, *coord_vectors, {}, xo, xc1, xc2);

    // Ensure logh_const is free
    logh_const.reset(nullptr);

    // setup a system of equations
    syst.reset(new System_of_eqs(*space, 0, ndom - 1));

    // the actual solution fields of the equations, i.e.
    // conformal factor, lapse, shift and (logarithmic) enthalpy
    // set this first before running syst_init()
    syst->add_var("H", *logh);

    // central enthalpy is a variable, fixed by the baryonic mass
    // integral
    syst->add_var("hc1", (*bconfig)(BCO_PARAMS::HC, NODES::BCO1));
    syst->add_var("hc2", (*bconfig)(BCO_PARAMS::HC, NODES::BCO2));

    Kadath::FUKA_EOS::set_eos_ope_struct<eos_t>()(*syst, eos_param);

    // populate all the boiler-plate constants, variables, and definitions
    syst_init();

    // center of mass on the x-axis connecting both companions
    // and orbital angular frequency parameter
    // in this case, both are fixed by the two central force-balance
    // equations
    syst->add_var("ome", (*bconfig)(BIN_PARAMS::GOMEGA));
    syst->add_var("xaxis", (*bconfig)(BIN_PARAMS::COM));

    // orbital rotation corrected by the "center of mass" shift, full shift
    binary_syst::add_orbital_shift(*syst, false);

    bool const corot = bconfig->control(CONTROLS::COROT_BIN);
    for (int d = 0; d < ndom; d++) {
        if ((d >= space->ADAPTED2 + 1) ||
            ((d >= space->ADAPTED1 + 1) && (d < space->NS2)))
            binary_syst::add_vacuum_eqdefs(*syst, d, corot, true);
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
                "integ(intS2) / Madm2 / Madm2 = chi2");
    }

    // force-balance equations at the center of each star
    // to fix the orbital frequency as well as the "center of mass"
    // (the latter can get very inaccurate in extreme configurations
    // with large residual linear momenta, for which the next stage is given)
    Index posori1(space->get_domain(space->NS1)->get_nbr_points());
    syst->add_eq_val(space->NS1, "ex^i * D_i H", posori1);

    Index posori2(space->get_domain(space->NS2)->get_nbr_points());
    syst->add_eq_val(space->NS2, "ex^i * D_i H", posori2);

    // add the first integral to get (approximate) hydrostatic equilibrium
    syst->add_eq_first_integral(space->NS1,
                                space->ADAPTED1,
                                "firstint",
                                "h - hc1");
    syst->add_eq_first_integral(space->NS2,
                                space->ADAPTED2,
                                "firstint",
                                "h - hc2");

    space->add_eq_int_volume(*syst,
                             space->NS1,
                             space->ADAPTED1,
                             "integvolume(intMb) = Mb1");

    space->add_eq_int_volume(*syst,
                             space->NS2,
                             space->ADAPTED2,
                             "integvolume(intMb) = Mb2");
}

/** @}*/
}  // namespace FUKA_Solvers
}  // namespace Kadath