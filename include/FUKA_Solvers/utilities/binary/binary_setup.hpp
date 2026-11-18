/**
 * @file binary_setup.hpp
 * @brief Pieces of the binary initial guess and regrid shared by the binary
 * solvers: domain radii, isolated companions read from file, and their
 * superposition.
 *
 * @copyright Copyright (c) 2026, GNU General Public Licensev3
 * This file is part of the KADATH library and published under
 * https://arxiv.org/abs/2103.09911
 */
#pragma once
#include <algorithm>
#include <cmath>
#include <initializer_list>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "Configurator/config_bco.hpp"
#include "Configurator/config_binary.hpp"
#include "EOS/EOS.hh"
#include "FUKA_Solvers/utilities/fields/scalar_calculations.hpp"
#include "bco_utilities.hpp"
#include "kadath_adapted.hpp"
#include "kadath_adapted_bh.hpp"

/** \addtogroup Solver_utils
 * \ingroup FUKA
 * @{*/
namespace Kadath::FUKA_Solvers::binary_setup {

/// Outer radius of both companions' shells: a third of the gap between the
/// larger companion and the midpoint of the binary.
template <class config_t>
void set_companion_outer_radii(config_t& bconfig) {
    double const rmax = std::max(bconfig(BCO_PARAMS::RMID, NODES::BCO1),
                                 bconfig(BCO_PARAMS::RMID, NODES::BCO2));
    double const rout_sep_est = (bconfig(BIN_PARAMS::DIST) / 2. - rmax) / 3. +
                                rmax;
    bconfig.set(BCO_PARAMS::ROUT, NODES::BCO1) = rout_sep_est;
    bconfig.set(BCO_PARAMS::ROUT, NODES::BCO2) = rout_sep_est;
}

/// Minimal shell thickness of both companions for a mass ratio q <= 1.
template <class config_t>
void set_min_shell_dr(config_t& bconfig, double const q) {
    double const dr = 1.7 *
                      std::pow(2.0, ::Kadath::bco_utils::shell_factor / q);
    bconfig.set(BCO_PARAMS::MIN_SHELL_DR, NODES::BCO1) = dr;
    bconfig.set(BCO_PARAMS::MIN_SHELL_DR, NODES::BCO2) = dr;
}

/// Bounds of the outer shells: REXT in steps of a quarter.
template <class config_t>
std::vector<double> outer_shell_bounds(config_t& bconfig) {
    std::vector<double> out_bounds(1 + bconfig(BIN_PARAMS::OUTER_SHELLS));
    for (std::size_t e = 0; e < out_bounds.size(); ++e)
        out_bounds[e] = bconfig(BIN_PARAMS::REXT) * (1. + e * 0.25);
    return out_bounds;
}

/**
 * @brief Coordinate radius of every collocation point, with the variable
 * outer boundary of the given outer-adapted domains; used to carry adapted
 * mappings to a new space (bco_utils::interp_adapted_mapping).
 */
template <class space_t>
Scalar radius_field(space_t const& space,
                    std::initializer_list<int> const outer_adapted_doms) {
    Scalar radius(space);
    radius.annule_hard();
    for (int d = 0; d < space.get_nbr_domains(); ++d)
        radius.set_domain(d) = space.get_domain(d)->get_radius();
    for (int d : outer_adapted_doms)
        radius.set_domain(d) =
            dynamic_cast<const Domain_shell_outer_adapted*>(space.get_domain(d))
                ->get_outer_radius();
    radius.std_base();
    return radius;
}

/// Domain bounds around a companion from the radial derivative of its
/// conformal factor (bco_utils::set_arb_boundsv3).
template <class config_t, class space_t>
std::vector<double> companion_bounds(config_t& bconfig,
                                     space_t& space,
                                     Scalar& conf,
                                     Vector const& shift,
                                     std::vector<int> const& excluded_doms,
                                     int const first_shell,
                                     NODES const bco,
                                     int const bound_dom = -1) {
    auto drPsi(compute_drPsi(space,
                             conf,
                             Metric_flat(space, shift.get_basis()),
                             excluded_doms,
                             bound_dom));
    return Kadath::bco_utils::set_arb_boundsv3(bconfig,
                                               drPsi,
                                               first_shell,
                                               bco);
}

/// Fields of an isolated companion read from its solution file; the black
/// hole has no matter fields.
struct Isolated_solution {
    /// Owner of the space (its concrete type's destructor is public, the
    /// base class's is not); the fields below refer to it.
    std::shared_ptr<void> space_owner;
    std::optional<Scalar> conf, lapse, logh, phi;
    std::optional<Vector> shift;
};

/**
 * @brief Isolated (boosted) neutron star: reads the NS_XCTS solution, sets
 * HC, NC and radii in its config, extends phi outside the star when present,
 * and builds its radius field for the adapted-mapping transfer.
 */
template <class eos_t>
struct Isolated_NS {
    Isolated_solution sol;
    std::optional<Scalar> radius;

    explicit Isolated_NS(kadath_config_boost<BCO_NS_INFO>& nsconfig) {
        using namespace ::Kadath::bco_utils;
        std::string const fname = nsconfig.space_filename();
        FILE* ff = fopen(fname.c_str(), "r");
        if (ff == nullptr)
            throw std::runtime_error("cannot open " + fname);
        auto space = std::make_shared<Space_spheric_adapted>(ff);
        sol.space_owner = space;
        sol.conf.emplace(*space, ff);
        sol.lapse.emplace(*space, ff);
        sol.shift.emplace(*space, ff);
        sol.logh.emplace(*space, ff);
        bool const has_phi = nsconfig.set_field(BCO_FIELDS::PHI);
        if (has_phi)
            sol.phi.emplace(*space, ff);
        fclose(ff);

        nsconfig.set(BCO_PARAMS::HC) =
            std::exp(get_boundary_val(0, *sol.logh, INNER_BC));
        nsconfig.set(BCO_PARAMS::NC) =
            EOS<eos_t, eos_var_t::DENSITY>::get(nsconfig(BCO_PARAMS::HC));
        update_config_NS_radii(*space, nsconfig, 1);

        if (has_phi) {
            auto const* inner = dynamic_cast<const Domain_shell_inner_adapted*>(
                space->get_domain(2));
            update_adapted_field(*sol.phi, 1, 2, inner, INNER_BC);
        }
        radius.emplace(radius_field(*space, {1}));
    }

    Space_spheric_adapted& space() const {
        return *std::static_pointer_cast<Space_spheric_adapted>(
            sol.space_owner);
    }

    /// Carry the star's surface mapping to the adapted shells of the binary.
    void map_surface(Kadath::Space const& binary, int const adapted) const {
        using namespace ::Kadath::bco_utils;
        interp_adapted_mapping(dynamic_cast<const Domain_shell_inner_adapted*>(
                                   binary.get_domain(adapted + 1)),
                               1,
                               *radius);
        interp_adapted_mapping(dynamic_cast<const Domain_shell_outer_adapted*>(
                                   binary.get_domain(adapted)),
                               1,
                               *radius);
    }
};

/**
 * @brief Isolated (boosted) black hole: reads the BH_XCTS solution, sets its
 * radii in its config, and extends the conformal factor and lapse into the
 * excised domain for the interpolation.
 */
struct Isolated_BH {
    Isolated_solution sol;

    explicit Isolated_BH(kadath_config_boost<BCO_BH_INFO>& bhconfig) {
        using namespace ::Kadath::bco_utils;
        std::string const fname = bhconfig.space_filename();
        FILE* ff = fopen(fname.c_str(), "r");
        if (ff == nullptr)
            throw std::runtime_error("cannot open " + fname);
        auto space = std::make_shared<Space_adapted_bh>(ff);
        sol.space_owner = space;
        sol.conf.emplace(*space, ff);
        sol.lapse.emplace(*space, ff);
        sol.shift.emplace(*space, ff);
        fclose(ff);
        update_config_BH_radii(*space, bhconfig, 1, *sol.conf);
    }

    Space_adapted_bh& space() const {
        return *std::static_pointer_cast<Space_adapted_bh>(sol.space_owner);
    }

    /// Extend the conformal factor and lapse (and the shift, if requested)
    /// from the horizon shell into the excised domain, so the interpolation
    /// near the horizon is smooth.
    void extend_into_excision(bool const with_shift = false) {
        using namespace ::Kadath::bco_utils;
        auto const* outer = dynamic_cast<const Domain_shell_outer_homothetic*>(
            space().get_domain(1));
        update_adapted_field(*sol.conf, 2, 1, outer, OUTER_BC);
        update_adapted_field(*sol.lapse, 2, 1, outer, OUTER_BC);
        if (with_shift)
            for (int i = 1; i <= 3; ++i)
                update_adapted_field(sol.shift->set(i), 2, 1, outer, OUTER_BC);
    }
};

/**
 * @brief Superposition of two isolated companions centered at xc1, xc2 on
 * the binary space, each weighted by exp(-r^4 / w^4) (w: DECAY), the
 * compactified domain set to its asymptotic values.
 */
struct Superposed_fields {
    Scalar conf, lapse, logh, phi;
    Vector shift;

    Superposed_fields(Kadath::Space const& space,
                      Base_tensor const& basis,
                      Isolated_solution const& co1,
                      Isolated_solution const& co2,
                      double const xc1,
                      double const xc2,
                      double const invw4_1,
                      double const invw4_2)
        : conf(space),
          lapse(space),
          logh(space),
          phi(space),
          shift(space, CON, basis) {
        logh.annule_hard();
        logh.std_base();
        conf.annule_hard();
        lapse.annule_hard();
        for (int i = 1; i <= 3; i++)
            shift.set(i).annule_hard();
        phi.annule_hard();

        int const ndom = space.get_nbr_domains();
        for (int dom = 0; dom < ndom; dom++) {
            Index pos(space.get_domain(dom)->get_nbr_points());
            do {
                double const x = space.get_domain(dom)->get_cart(1)(pos);
                double const y = space.get_domain(dom)->get_cart(2)(pos);
                double const z = space.get_domain(dom)->get_cart(3)(pos);
                if (dom == ndom - 1) {
                    // the outer collocation point is at infinity
                    conf.set_domain(dom).set(pos) = 1.;
                    lapse.set_domain(dom).set(pos) = 1.;
                    continue;
                }
                double const r2 = y * y + z * z;
                auto at = [&](double const xc, double const invw4) {
                    Point p(3);
                    p.set(1) = x - xc;
                    p.set(2) = y;
                    p.set(3) = z;
                    double const rsq = (x - xc) * (x - xc) + r2;
                    return std::make_pair(p, std::exp(-rsq * rsq * invw4));
                };
                auto const [p1, w1] = at(xc1, invw4_1);
                auto const [p2, w2] = at(xc2, invw4_2);

                conf.set_domain(dom).set(pos) =
                    1. + w1 * (co1.conf->val_point(p1) - 1.) +
                    w2 * (co2.conf->val_point(p2) - 1.);
                lapse.set_domain(dom).set(pos) =
                    1. + w1 * (co1.lapse->val_point(p1) - 1.) +
                    w2 * (co2.lapse->val_point(p2) - 1.);

                double h = 0.;
                if (co1.logh)
                    h += w1 * co1.logh->val_point(p1);
                if (co2.logh)
                    h += w2 * co2.logh->val_point(p2);
                logh.set_domain(dom).set(pos) = h;

                double ph = 0.;
                if (co1.phi)
                    ph += w1 * co1.phi->val_point(p1);
                if (co2.phi)
                    ph += w2 * co2.phi->val_point(p2);
                phi.set_domain(dom).set(pos) = ph;

                for (int i = 1; i <= 3; i++)
                    shift.set(i).set_domain(dom).set(pos) =
                        w1 * (*co1.shift)(i).val_point(p1) +
                        w2 * (*co2.shift)(i).val_point(p2);
            } while (pos.inc());
        }
    }

    /// Standard spectral bases of all fields.
    void std_base() {
        conf.std_base();
        lapse.std_base();
        logh.std_base();
        shift.std_base();
        phi.std_base();
    }
};

}  // namespace Kadath::FUKA_Solvers::binary_setup

/** @}*/
