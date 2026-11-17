/**
 * @file binary_syst.hpp
 * @brief System-of-equations building blocks shared by the binary solvers
 * with neutron stars (BNS, BHNS): coordinate fields, stellar spin, matter
 * definitions, orbital shift, and the constraint, Euler and
 * velocity-potential equations in matter and vacuum domains.
 *
 * Equations: arXiv:2103.09911 (FUKA); inside the stars the constraint
 * equations are multiplied by delta = p / rho.
 *
 * @copyright Copyright (c) 2026, GNU General Public Licensev3
 * This file is part of the KADATH library and published under
 * https://arxiv.org/abs/2103.09911
 */
#pragma once
#include <cmath>
#include <iostream>
#include <string>

#include "Configurator/config_binary.hpp"
#include "bco_utilities.hpp"
#include "coord_fields.hpp"
#include "kadath.hpp"

/** \addtogroup Solver_utils
 * \ingroup FUKA
 * @{*/
namespace Kadath::FUKA_Solvers::binary_syst {

/// Coordinate vector fields of a binary: rotation fields, Cartesian basis,
/// surface normals of both companions and at infinity.
template <class cfary_t>
void add_coordinate_fields(System_of_eqs& syst, cfary_t& coord_vectors) {
    syst.add_cst("mg", *coord_vectors[GLOBAL_ROT]);
    syst.add_cst("mm", *coord_vectors[BCO1_ROT]);
    syst.add_cst("mp", *coord_vectors[BCO2_ROT]);

    syst.add_cst("ex", *coord_vectors[EX]);
    syst.add_cst("ey", *coord_vectors[EY]);
    syst.add_cst("ez", *coord_vectors[EZ]);

    syst.add_cst("sm", *coord_vectors[S_BCO1]);
    syst.add_cst("sp", *coord_vectors[S_BCO2]);
    syst.add_cst("einf", *coord_vectors[S_INF]);
}

/**
 * @brief Stellar rotation: zero for corotation, otherwise the velocity
 * potential phi plus a rigid spin omes1/omes2 per companion, fixed
 * (FIXED_BCOMEGA) or solved for the configured chi.
 *
 * @return true for corotation
 */
template <class config_t>
bool add_spin_parameters(System_of_eqs& syst, config_t& bconfig, Scalar& phi) {
    if (bconfig.control(CONTROLS::COROT_BIN)) {
        bconfig.set(BCO_PARAMS::OMEGA, NODES::BCO1) = 0.;
        syst.add_cst("omes1", bconfig(BCO_PARAMS::OMEGA, NODES::BCO1));
        bconfig.set(BCO_PARAMS::OMEGA, NODES::BCO2) = 0.;
        syst.add_cst("omes2", bconfig(BCO_PARAMS::OMEGA, NODES::BCO2));
        return true;
    }
    syst.add_var("phi", phi);

    auto add_omega = [&](NODES const bco, char const* name) {
        if (!std::isnan(bconfig.set(BCO_PARAMS::FIXED_BCOMEGA, bco))) {
            syst.add_cst(name, bconfig(BCO_PARAMS::FIXED_BCOMEGA, bco));
            bconfig.set(BCO_PARAMS::OMEGA, bco) =
                bconfig(BCO_PARAMS::FIXED_BCOMEGA, bco);
        } else {
            syst.add_var(name, bconfig(BCO_PARAMS::OMEGA, bco));
        }
    };
    add_omega(NODES::BCO1, "omes1");
    add_omega(NODES::BCO2, "omes2");
    return false;
}

/// Spin velocity field s^i = omes * m^i and eta_i in the star's domains.
inline void add_star_spin_defs(System_of_eqs& syst,
                               int const dmin,
                               int const dmax,
                               std::string const& omes,
                               std::string const& m) {
    std::string const s{"s^i  = " + omes + " * " + m + "^i"};
    for (int d = dmin; d <= dmax; ++d) {
        syst.add_def(d, s.c_str());
        syst.add_def(d, "eta_i  = D_i phi + P^4 * s_i");
    }
}

/// Enthalpy, EOS quantities and delta = p / rho from the log enthalpy H.
inline void add_matter_defs(System_of_eqs& syst) {
    syst.add_def("h = exp(H)");
    syst.add_def("rho = rho(h)");
    syst.add_def("eps = eps(h)");
    syst.add_def("press = press(h)");
    syst.add_def("dHdlnrho = dHdlnrho(h)");
    syst.add_def("delta = h - eps - 1.");
}

/// ADM linear momentum integrands at infinity.
inline void add_momentum_integrands(System_of_eqs& syst, int const ndom) {
    syst.add_def(ndom - 1, "intPx = A_i^j * ex_j * einf^i");
    syst.add_def(ndom - 1, "intPy = A_i^j * ey_j * einf^i");
    syst.add_def(ndom - 1, "intPz = A_i^j * ez_j * einf^i");
}

/**
 * @brief Orbital frequency and radial velocity of the eccentricity reduction:
 * 3.5PN estimates when ADOT or ECC_OMEGA is unset or USE_PN is on, otherwise
 * the configured ECC_OMEGA.
 *
 * @param M1, M2 companion masses for the PN estimate
 */
template <class config_t>
void set_ecc_red_orbit(config_t& bconfig,
                       int const rank,
                       double const M1,
                       double const M2) {
    if (std::isnan(bconfig.set(BIN_PARAMS::ADOT)) ||
        std::isnan(bconfig.set(BIN_PARAMS::ECC_OMEGA)) ||
        bconfig.control(CONTROLS::USE_PN)) {
        ::Kadath::bco_utils::KadathPNOrbitalParams(bconfig, M1, M2);
        if (rank == 0)
            std::cout << "### Using PN estimate for adot and omega! ###"
                      << std::endl;
        bconfig.set(BIN_PARAMS::ECC_OMEGA) = bconfig(BIN_PARAMS::GOMEGA);
    } else {
        bconfig.set(BIN_PARAMS::GOMEGA) = bconfig(BIN_PARAMS::ECC_OMEGA);
    }
}

/**
 * @brief Orbital shift B^i = beta^i + ome * Morb^i (+ adot * comr^i for the
 * eccentricity reduction), Morb corrected by the center-of-mass shifts
 * xaxis, yaxis.
 *
 * @param with_yaxis include the y shift of the center of mass
 * @param adot radial velocity (eccentricity reduction) or nullptr
 * @param cart position vector field, required with adot
 */
inline void add_orbital_shift(System_of_eqs& syst,
                              bool const with_yaxis,
                              double* adot = nullptr,
                              Vector* cart = nullptr) {
    syst.add_def(with_yaxis ? "Morb^i = mg^i + xaxis * ey^i + yaxis * ex^i"
                            : "Morb^i = mg^i + xaxis * ey^i");
    std::string bigB{"B^i = bet^i + ome * Morb^i"};
    if (adot != nullptr) {
        syst.add_cst("adot", *adot);
        syst.add_cst("r", *cart);
        syst.add_def("comr^i = r^i - xaxis * ex^i + yaxis * ey^i");
        bigB += " + adot * comr^i";
    }
    syst.add_def(bigB.c_str());
}

/**
 * @brief Vacuum domain: source-free constraint equations; no velocity
 * potential (unless corotating) and, if requested, no matter.
 */
inline void add_vacuum_eqdefs(System_of_eqs& syst,
                              int const d,
                              bool const corot,
                              bool const zero_enthalpy) {
    if (!corot)
        syst.add_eq_full(d, "phi= 0");
    if (zero_enthalpy)
        syst.add_eq_full(d, "H  = 0");

    syst.add_def(d, "eqP = D^i D_i P + A_ij * A^ij / P^7 / 8");
    syst.add_def(d, "eqNP = D^i D_i NP - 7. / 8. * NP / P^8 * A_ij * A^ij");
    syst.add_def(d,
                 "eqbet^i = D_j D^j bet^i + D^i D_j bet^j / 3. - 2. * A^ij * "
                 "D_j Ntilde");
}

/**
 * @brief Matter domain: fluid velocity and first integral of the Euler
 * equation (corotating or irrotational + spin with velocity potential),
 * sources, constraint equations multiplied by delta, mass integrands.
 */
inline void add_matter_eqdefs(System_of_eqs& syst,
                              int const d,
                              bool const corot) {
    if (corot) {
        syst.add_def(d, "U^i    = B^i / N");
        syst.add_def(d, "Usquare= P^4 * U_i * U^i");
        syst.add_def(d, "Wsquare= 1 / (1 - Usquare)");
        syst.add_def(d, "W      = sqrt(Wsquare)");
        syst.add_def(d, "firstint = log(h * N / W)");
    } else {
        syst.add_def(d, "Wsquare= eta^i * eta_i / h^2 / P^4 + 1.");
        syst.add_def(d, "W      = sqrt(Wsquare)");
        syst.add_def(d, "U^i    = eta^i / P^4 / h / W");
        syst.add_def(d, "Usquare= P^4 * U_i * U^i");
        syst.add_def(d, "V^i    = N * U^i - B^i");
        syst.add_def(d, "firstint = log(h * N / W + D_i phi * V^i)");
        syst.add_def(d,
                     "eqphi  = P^6 * W * V^i * D_i H + dHdlnrho * D_i "
                     "(P^6 * W * V^i)");
    }
    syst.add_def(d, "Etilde = press * h * Wsquare - press * delta");
    syst.add_def(d,
                 "Stilde = 3 * press * delta + (Etilde + press * "
                 "delta) * Usquare");
    syst.add_def(d, "ptilde^i = press * h * Wsquare * U^i");

    syst.add_def(d,
                 "eqP = delta * D^i D_i P + A_ij * A^ij / P^7 / 8 * delta "
                 "+ 4piG / 2. * P^5 * Etilde");
    syst.add_def(d,
                 "eqNP = delta * D^i D_i NP - 7. / 8. * NP / P^8 * delta * "
                 "A_ij *A^ij "
                 "- 4piG / 2. * N * P^5 * (Etilde + 2. * Stilde)");
    syst.add_def(d,
                 "eqbet^i= delta * D_j D^j bet^i + delta * D^i D_j bet^j / 3. "
                 "- 2. * delta * A^ij * D_j Ntilde - 4. * 4piG * N * P^4 * "
                 "ptilde^i");

    syst.add_def(d, "intMb  = P^6 * rho * W");
    syst.add_def(d, "intM   = - D_i D^i P * 2. / 4piG");
}

/// Velocity-potential equation in a star's domains, matched across the
/// inner domain boundaries.
inline void add_velocity_potential_eqs(System_of_eqs& syst,
                                       int const nucleus,
                                       int const adapted) {
    for (int i = nucleus; i < adapted; ++i) {
        syst.add_eq_vel_pot(i, 2, "eqphi = 0", "phi=0");
        syst.add_eq_matching(i, OUTER_BC, "phi");
        syst.add_eq_matching(i, OUTER_BC, "dn(phi)");
    }
    syst.add_eq_vel_pot(adapted, 2, "eqphi = 0", "phi=0");
}

/// Metric equations with continuity of their normal derivatives, and
/// asymptotic flatness at infinity.
template <class space_t>
void add_metric_equations(System_of_eqs& syst, space_t& space, int const ndom) {
    space.add_eq(syst, "eqNP= 0", "N", "dn(N)");
    space.add_eq(syst, "eqP= 0", "P", "dn(P)");
    space.add_eq(syst, "eqbet^i= 0", "bet^i", "dn(bet^i)");

    syst.add_eq_bc(ndom - 1, OUTER_BC, "N=1");
    syst.add_eq_bc(ndom - 1, OUTER_BC, "P=1");
    syst.add_eq_bc(ndom - 1, OUTER_BC, "bet^i=0");
}

}  // namespace Kadath::FUKA_Solvers::binary_syst

/** @}*/
