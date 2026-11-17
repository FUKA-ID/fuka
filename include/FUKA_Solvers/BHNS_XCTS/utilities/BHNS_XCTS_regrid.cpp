#include <algorithm>
#include <vector>

#include "FUKA_Solvers/utilities/binary/binary_setup.hpp"
#include "Solvers/fuka_syst/fuka_syst_tools.hpp"

namespace Kadath::FUKA_Solvers {

inline void BHNS_XCTS_base::regrid() {
    if (static_cast<int>((*bconfig)(BIN_PARAMS::BIN_RES)) % 2 == 0)
        throw std::invalid_argument(
            "BHNS_XCTS::regrid: the resolution must be odd (9, 11, 13, ...)");
    regrid_to("bhns_regrid");
}

/**
 * Domain bounds from the current solution: star and horizon radii, the
 * companions' shells from the radial derivative of the conformal factor; the
 * fields are imported into the new space with the star's surface mapping
 * carried over and no matter outside the star or inside the horizon.
 */
inline void BHNS_XCTS_base::write_regridded_solution(
    base_config_t& target) const {
    using namespace ::Kadath::bco_utils;
    using namespace binary_setup;

    target.set(BIN_PARAMS::Q) = target(BCO_PARAMS::MADM, NODES::BCO1) /
                                target(BCO_PARAMS::MCH, NODES::BCO2);
    if (target.set(BIN_PARAMS::Q) > 1.0)
        target.set(BIN_PARAMS::Q) = 1.0 / target.set(BIN_PARAMS::Q);
    set_min_shell_dr(target, target(BIN_PARAMS::Q));

    if (std::isnan(target.set(BIN_PARAMS::OUTER_SHELLS)))
        target.set(BIN_PARAMS::OUTER_SHELLS) = 0;

    // fields are extended across the star surface and into the excision
    // before the import; work on copies so the solver keeps its state
    Scalar old_conf(*conformal_factor);
    Scalar old_lapse(*lapse);
    Vector old_shift(*shift);
    Scalar old_phi(*phi);

    if (!target.control(CONTROLS::USE_CONFIG_VARS)) {
        update_config_NS_radii(*space, target, space->ADAPTEDNS, NODES::BCO1);
        update_config_BH_radii(*space,
                               target,
                               space->ADAPTEDBH,
                               old_conf,
                               NODES::BCO2);
        set_companion_outer_radii(target);
    }

    Scalar const old_space_radius{
        radius_field(*space, {space->ADAPTEDNS, space->ADAPTEDBH})};
    std::vector<double> const out_bounds{outer_shell_bounds(target)};

    std::vector<int> exclusion_doms{
        FUKA_Syst_tools::vector_of_domains(space->NS, space->ADAPTEDNS)};
    exclusion_doms.push_back(space->BH);
    exclusion_doms.push_back(space->BH + 1);
    auto drPsi(compute_drPsi(*space,
                             old_conf,
                             Metric_flat(*space, shift->get_basis()),
                             exclusion_doms,
                             space->OUTER));
    std::vector<double> const NS_bounds{
        set_arb_boundsv3(target, drPsi, space->ADAPTEDNS + 1, NODES::BCO1)};
    std::vector<double> const BH_bounds{
        set_arb_boundsv3(target, drPsi, space->ADAPTEDBH + 1, NODES::BCO2)};

    std::cout << "Local bounds:" << std::endl;
    print_bounds("NS-bounds", NS_bounds);
    print_bounds("BH-bounds", BH_bounds);
    print_bounds("outer-bounds", out_bounds);
    std::cout << std::endl;

    Space_bhns new_space(space->get_type_base(),
                         target(BIN_PARAMS::DIST),
                         NS_bounds,
                         BH_bounds,
                         out_bounds,
                         target(BIN_PARAMS::BIN_RES),
                         target(BCO_PARAMS::NINSHELLS, NODES::BCO1));
    Base_tensor new_basis(new_space, CARTESIAN_BASIS);

    std::cout << "Resolution of old space: ";
    print_constant_space_resolution(*space);
    std::cout << "Resolution of new space: ";
    print_constant_space_resolution(new_space);
    std::cout << "\nOld bounds:" << std::endl;
    print_bounds_from_space(*space);
    std::cout << "New bounds:" << std::endl;
    print_bounds_from_space(new_space);

    auto const* old_bh_outer =
        dynamic_cast<const Domain_shell_outer_homothetic*>(
            space->get_domain(space->ADAPTEDBH));
    update_adapted_field(old_conf,
                         space->ADAPTEDBH + 1,
                         space->ADAPTEDBH,
                         old_bh_outer,
                         OUTER_BC);
    update_adapted_field(old_lapse,
                         space->ADAPTEDBH + 1,
                         space->ADAPTEDBH,
                         old_bh_outer,
                         OUTER_BC);
    for (int i = 1; i < 4; ++i)
        update_adapted_field(old_shift.set(i),
                             space->ADAPTEDBH + 1,
                             space->ADAPTEDBH,
                             old_bh_outer,
                             OUTER_BC);
    update_adapted_field(old_phi,
                         space->ADAPTEDNS,
                         space->ADAPTEDNS + 1,
                         dynamic_cast<const Domain_shell_inner_adapted*>(
                             space->get_domain(space->ADAPTEDNS + 1)),
                         INNER_BC);

    interp_adapted_mapping(dynamic_cast<const Domain_shell_inner_adapted*>(
                               new_space.get_domain(new_space.ADAPTEDNS + 1)),
                           space->ADAPTEDNS,
                           old_space_radius);
    interp_adapted_mapping(dynamic_cast<const Domain_shell_outer_adapted*>(
                               new_space.get_domain(new_space.ADAPTEDNS)),
                           space->ADAPTEDNS,
                           old_space_radius);

    Scalar new_conf(new_space);
    new_conf = 1.;
    Scalar new_lapse(new_space);
    new_lapse = 1.;
    Vector new_shift(new_space, CON, new_basis);
    for (int i = 1; i <= 3; i++)
        new_shift.set(i).annule_hard();
    Scalar new_logh(new_space);
    new_logh.annule_hard();
    Scalar new_phi(new_space);
    new_phi.annule_hard();

    new_conf.import(old_conf);
    new_lapse.import(old_lapse);
    new_logh.import(*logh);
    new_phi.import(old_phi);
    for (int i = 1; i <= 3; ++i)
        new_shift.set(i).import(old_shift.set(i));

    // no matter or velocity potential outside the star, nothing inside the
    // horizon
    new_logh.set_domain(new_space.ADAPTEDNS + 1).annule_hard();
    new_phi.set_domain(new_space.ADAPTEDNS + 1).annule_hard();
    for (int d = new_space.BH; d < new_space.get_nbr_domains(); ++d) {
        new_logh.set_domain(d).annule_hard();
        new_phi.set_domain(d).annule_hard();
        if (d < new_space.BH + 2) {
            new_conf.set_domain(d).annule_hard();
            new_lapse.set_domain(d).annule_hard();
            for (int i = 1; i <= 3; i++)
                new_shift.set(i).set_domain(d).annule_hard();
        }
    }

    new_lapse.std_base();
    new_conf.std_base();
    new_logh.std_base();
    new_shift.std_base();
    new_phi.std_base();

    save_to_file(new_space,
                 target,
                 new_conf,
                 new_lapse,
                 new_shift,
                 new_logh,
                 new_phi);
}

}  // namespace Kadath::FUKA_Solvers
