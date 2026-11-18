#include <vector>

#include "FUKA_Solvers/utilities/binary/binary_setup.hpp"

namespace Kadath::FUKA_Solvers {

inline void BBH_XCTS::regrid() {
    if (static_cast<int>((*bconfig)(BIN_PARAMS::BIN_RES)) % 2 == 0)
        throw std::invalid_argument(
            "BBH_XCTS::regrid: the resolution must be odd (9, 11, 13, ...)");
    regrid_to("bbh_regrid");
}

/**
 * Domain bounds from the current solution: horizon radii, the holes' shells
 * from the radial derivative of the conformal factor; the fields are imported
 * into the new space, nothing inside the horizons.
 */
inline void BBH_XCTS::write_regridded_solution(base_config_t& target) const {
    using namespace ::Kadath::bco_utils;
    using namespace binary_setup;

    // fields are extended into the excised domains before the import; work
    // on copies so the solver keeps its state
    Scalar old_conf(*conformal_factor);
    Scalar old_lapse(*lapse);
    Vector old_shift(*shift);

    if (!target.control(CONTROLS::USE_CONFIG_VARS)) {
        update_config_BH_radii(*space,
                               target,
                               space->BH1 + 1,
                               old_conf,
                               NODES::BCO1);
        update_config_BH_radii(*space,
                               target,
                               space->BH2 + 1,
                               old_conf,
                               NODES::BCO2);
    }
    if (std::isnan(target.set(BIN_PARAMS::OUTER_SHELLS)))
        target.set(BIN_PARAMS::OUTER_SHELLS) = 0;
    target.set(BIN_PARAMS::Q) = target(BCO_PARAMS::MCH, NODES::BCO1) /
                                target(BCO_PARAMS::MCH, NODES::BCO2);
    if (target.set(BIN_PARAMS::Q) > 1.0)
        target.set(BIN_PARAMS::Q) = 1.0 / target.set(BIN_PARAMS::Q);
    if (!target.control(CONTROLS::USE_CONFIG_VARS)) {
        target.set(BIN_PARAMS::REXT) = 2. * target(BIN_PARAMS::DIST);
        set_companion_outer_radii(target);
        set_min_shell_dr(target, target(BIN_PARAMS::Q));
    }

    std::vector<double> const out_bounds{outer_shell_bounds(target)};
    std::vector<int> const exclusion_doms{space->BH1,
                                          space->BH1 + 1,
                                          space->BH2,
                                          space->BH2 + 1};
    auto drPsi(compute_drPsi(*space,
                             old_conf,
                             Metric_flat(*space, shift->get_basis()),
                             exclusion_doms,
                             space->OUTER));
    std::vector<double> const BH1_bounds{
        set_arb_boundsv3(target, drPsi, space->BH1 + 2, NODES::BCO1)};
    std::vector<double> const BH2_bounds{
        set_arb_boundsv3(target, drPsi, space->BH2 + 2, NODES::BCO2)};

    Space_bin_bh new_space(space->get_type_base(),
                           target(BIN_PARAMS::DIST),
                           BH1_bounds,
                           BH2_bounds,
                           out_bounds,
                           target(BIN_PARAMS::BIN_RES));
    Base_tensor new_basis(new_space, CARTESIAN_BASIS);

    std::cout << "Resolution of old space: ";
    print_constant_space_resolution(*space);
    std::cout << "Resolution of new space: ";
    print_constant_space_resolution(new_space);
    std::cout << "\nold bounds:" << std::endl;
    print_bounds_from_space(*space);
    std::cout << "New bounds:" << std::endl;
    print_bounds_from_space(new_space);

    for (int const nuc : {space->BH1, space->BH2}) {
        auto const* outer = dynamic_cast<const Domain_shell_outer_homothetic*>(
            space->get_domain(nuc + 1));
        update_adapted_field(old_conf, nuc + 2, nuc + 1, outer, OUTER_BC);
        update_adapted_field(old_lapse, nuc + 2, nuc + 1, outer, OUTER_BC);
        for (int j = 1; j < 4; ++j)
            update_adapted_field(old_shift.set(j),
                                 nuc + 2,
                                 nuc + 1,
                                 outer,
                                 OUTER_BC);
    }

    Scalar new_conf(new_space);
    new_conf = 1.;
    Scalar new_lapse(new_space);
    new_lapse = 1.;
    Vector new_shift(new_space, CON, new_basis);
    for (int i = 1; i <= 3; i++)
        new_shift.set(i).annule_hard();

    new_conf.import(old_conf);
    new_lapse.import(old_lapse);
    for (int c = 1; c <= 3; ++c)
        new_shift.set(c).import(old_shift(c));

    for (int const nuc : {new_space.BH1, new_space.BH2})
        for (int const d : {nuc, nuc + 1}) {
            new_lapse.set_domain(d).annule_hard();
            new_conf.set_domain(d).annule_hard();
            for (int i = 1; i <= 3; ++i)
                new_shift.set(i).set_domain(d).annule_hard();
        }

    new_lapse.std_base();
    new_conf.std_base();
    new_shift.std_base();

    save_to_file(new_space, target, new_conf, new_lapse, new_shift);
}

}  // namespace Kadath::FUKA_Solvers
