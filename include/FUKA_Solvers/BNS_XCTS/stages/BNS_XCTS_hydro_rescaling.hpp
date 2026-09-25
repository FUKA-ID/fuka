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
    if (solver_stage == STAGES::ECC_RED) {
        // determine whether to use PN estimates of the orbital frequency and adot
        // in case of the eccentricity stage
        if (std::isnan(bconfig->set(BIN_PARAMS::ADOT)) ||
            std::isnan(bconfig->set(BIN_PARAMS::ECC_OMEGA)) ||
            bconfig->control(CONTROLS::USE_PN)) {
            ::Kadath::bco_utils::KadathPNOrbitalParams(
                *bconfig,
                (*bconfig)(BCO_PARAMS::MADM, NODES::BCO1),
                (*bconfig)(BCO_PARAMS::MADM, NODES::BCO2));

            if (rank == 0)
                std::cout << "### Using PN estimate for adot and omega! ###"
                          << std::endl;
            bconfig->set(BIN_PARAMS::ECC_OMEGA) =
                (*bconfig)(BIN_PARAMS::GOMEGA);
        } else {
            bconfig->set(BIN_PARAMS::GOMEGA) =
                (*bconfig)(BIN_PARAMS::ECC_OMEGA);
        }
    }

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

    Param p;
    Kadath::FUKA_EOS::set_eos_ope_struct<eos_t>()(*syst, p);

    // populate all the boiler-plate constants, variables, and definitions
    syst_init();

    bool fixed_com = this->solver_stage != STAGES::ECC_RED;
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

    // orbital rotation vector field, corrected by the "center of mass" shift
    syst->add_def("Morb^i = mg^i + xaxis * ey^i + yaxis * ex^i");

    std::string bigB{"B^i = bet^i + ome * Morb^i"};
    if (solver_stage == STAGES::ECC_RED) {
        syst->add_cst("adot", (*bconfig)(BIN_PARAMS::ADOT));
        syst->add_cst("r", CART);
        syst->add_def("comr^i = r^i - xaxis * ex^i + yaxis * ey^i");
        // add contribution of ADOT to the total shift definition
        bigB += " + adot * comr^i";
    }

    // full shift vector including inertial + orbital contributions
    syst->add_def(bigB.c_str());

    // the actual equations, defined differently in the different domains
    for (int d = 0; d < ndom; d++) {
        // if outside the stellar domains, without matter sources
        // resort to the source-free constraint equations
        // and set matter (and velocity potential) to zero
        if ((d >= space->ADAPTED2 + 1) ||
            ((d >= space->ADAPTED1 + 1) && (d < space->NS2))) {
            if (!bconfig->control(COROT_BIN))
                syst->add_eq_full(d, "phi= 0");

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
                "eqP = delta * D^i D_i P + A_ij * A^ij / P^7 / 8 * delta "
                "+ 4piG / 2. * P^5 * Etilde");
            syst->add_def(
                d,
                "eqNP = delta * D^i D_i NP - 7. / 8. * NP / P^8 * delta * "
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