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
    // We use `config_filename()` vs `config_filename_abs()` since
    // `solution_exists` will probe the HOME_KADATH/COs directory
    auto const current = bconfig->config_filename();
    if (!bconfig->control(RESOLVE) && solution_exists()) {
        if (rank == 0) {
            std::cout << "Solved previously: " << bconfig->config_filename_abs()
                      << std::endl;
        }
        reload();
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

    Param p;
    Kadath::FUKA_EOS::set_eos_ope_struct<eos_t>()(*syst, p);

    // populate all the boiler-plate constants, variables, and definitions
    syst_init();

    // center of mass on the x-axis connecting both companions
    // and orbital angular frequency parameter
    // in this case, both are fixed by the two central force-balance
    // equations
    syst->add_var("ome", (*bconfig)(BIN_PARAMS::GOMEGA));
    syst->add_var("xaxis", (*bconfig)(BIN_PARAMS::COM));

    // orbital rotation vector field, corrected by the "center of mass" shift
    syst->add_def("Morb^i = mg^i + xaxis * ey^i");

    // full shift vector field, incoporating the orbital part
    syst->add_def("B^i= bet^i + ome * Morb^i");

    // the actual equations, defined differently in the different domains
    for (int d = 0; d < ndom; d++) {
        // if outside the stellar domains, without matter sources
        // resort to the source-free constraint equations
        // and set matter (and velocity potential) to zero
        if ((d >= space->ADAPTED2 + 1) ||
            ((d >= space->ADAPTED1 + 1) && (d < space->NS2))) {
            if (!bconfig->control(CONTROLS::COROT_BIN))
                syst->add_eq_full(d, "phi= 0");

            syst->add_eq_full(d, "H  = 0");
            syst->add_def(d, "eqP = D^i D_i P + A_ij * A^ij / P^7 / 8");
            syst->add_def(
                d,
                "eqNP = D^i D_i NP - 7. / 8. * NP / P^8 * A_ij * A^ij");
            syst->add_def(
                d,
                "eqbet^i = D_j D^j bet^i + D^i D_j bet^j / 3. - 2. * A^ij * "
                "D_j Ntilde");

        }
        // in case of the domains harboring the two stars
        // define all derived matter related quantities
        // and enforce the transformed contraint equations (i.e. multiplied by p /
        // rho)
        else {
            // 3-velocity and first integral of the Euler equation
            // in case of corotation
            if (bconfig->control(CONTROLS::COROT_BIN)) {
                syst->add_def(d, "U^i    = B^i / N");
                syst->add_def(d, "Usquare= P^4 * U_i * U^i");
                syst->add_def(d, "Wsquare= 1 / (1 - Usquare)");
                syst->add_def(d, "W      = sqrt(Wsquare)");
                syst->add_def(d, "firstint = log(h * N / W)");
            }
            // 3-velocity, (approximate) first integral of the Euler equation
            // and velocity potential equation
            // in case of irrotational or spinning companions
            else {
                syst->add_def(d, "Wsquare= eta^i * eta_i / h^2 / P^4 + 1.");
                syst->add_def(d, "W      = sqrt(Wsquare)");
                syst->add_def(d, "U^i    = eta^i / P^4 / h / W");
                syst->add_def(d, "Usquare= P^4 * U_i * U^i");
                syst->add_def(d, "V^i    = N * U^i - B^i");
                syst->add_def(d, "firstint = log(h * N / W + D_i phi * V^i)");
                syst->add_def(d,
                              "eqphi  = P^6 * W * V^i * D_i H + dHdlnrho * D_i "
                              "(P^6 * W * V^i)");
            }
            // transformed source terms
            syst->add_def(d, "Etilde = press * h * Wsquare - press * delta");
            syst->add_def(d,
                          "Stilde = 3 * press * delta + (Etilde + press * "
                          "delta) * Usquare");
            syst->add_def(d, "ptilde^i = press * h * Wsquare * U^i");

            // transformed constraint equations
            syst->add_def(
                d,
                "eqP    = delta * D^i D_i P + A_ij * A^ij / P^7 / 8 * delta "
                "+ 4piG / 2. * P^5 * Etilde");
            syst->add_def(
                d,
                "eqNP   = delta * D^i D_i NP - 7. / 8. * NP / P^8 * delta * "
                "A_ij *A^ij "
                "- 4piG / 2. * N * P^5 * (Etilde + 2. * Stilde)");
            syst->add_def(
                d,
                "eqbet^i= delta * D_j D^j bet^i + delta * D^i D_j bet^j / 3. "
                "- 2. * delta * A^ij * D_j Ntilde - 4. * 4piG * N * P^4 * "
                "ptilde^i");

            // baryonic mass volume integrant
            syst->add_def(d, "intMb  = P^6 * rho * W");
            // quasi-local ADM mass volume integrant
            syst->add_def(d, "intM   = - D_i D^i P * 2. / 4piG");
        }
    }
    // add equations to the system and demand continuity
    // along the normal of the domains
    space->add_eq(*syst, "eqNP= 0", "N", "dn(N)");
    space->add_eq(*syst, "eqP= 0", "P", "dn(P)");
    space->add_eq(*syst, "eqbet^i= 0", "bet^i", "dn(bet^i)");

    // boundary conditions at infinity
    syst->add_eq_bc(ndom - 1, OUTER_BC, "N=1");
    syst->add_eq_bc(ndom - 1, OUTER_BC, "P=1");
    syst->add_eq_bc(ndom - 1, OUTER_BC, "bet^i=0");

    // boundary conditions defining the boundary of the adapted domains, i.e.
    // vanishing matter
    syst->add_eq_bc(space->ADAPTED1, OUTER_BC, "H = 0");
    syst->add_eq_bc(space->ADAPTED2, OUTER_BC, "H = 0");

    // determine equations to add based on corotation
    // or mixed spin binaries based on chi or fixed omega
    if (!bconfig->control(CONTROLS::COROT_BIN)) {
        syst->add_eq_bc(space->ADAPTED1, OUTER_BC, "V^i * D_i H = 0");
        syst->add_eq_bc(space->ADAPTED2, OUTER_BC, "V^i * D_i H = 0");

        for (int i = space->NS1; i < space->ADAPTED1; ++i) {
            syst->add_eq_vel_pot(i, 2, "eqphi = 0", "phi=0");
            syst->add_eq_matching(i, OUTER_BC, "phi");
            syst->add_eq_matching(i, OUTER_BC, "dn(phi)");
        }
        syst->add_eq_vel_pot(space->ADAPTED1, 2, "eqphi = 0", "phi=0");

        for (int i = space->NS2; i < space->ADAPTED2; ++i) {
            syst->add_eq_vel_pot(i, 2, "eqphi = 0", "phi=0");
            syst->add_eq_matching(i, OUTER_BC, "phi");
            syst->add_eq_matching(i, OUTER_BC, "dn(phi)");
        }
        syst->add_eq_vel_pot(space->ADAPTED2, 2, "eqphi = 0", "phi=0");

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