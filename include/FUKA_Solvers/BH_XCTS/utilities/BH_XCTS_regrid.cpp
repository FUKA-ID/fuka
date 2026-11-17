#include "FUKA_Solvers/BH_XCTS/utilities/BH_XCTS_setup.hpp"

namespace Kadath::FUKA_Solvers {

inline void BH_XCTS_base::regrid() {
    if (static_cast<int>((*bconfig)(BCO_RES)) % 2 == 0)
        throw std::invalid_argument(
            "BH_XCTS::regrid: the resolution must be odd (9, 11, 13, ...)");
    regrid_to("bh_regrid");
}

/**
 * Domain bounds from the current solution: the excision radius from the
 * conformal factor on the horizon, r ~ M / psi^2 (arXiv:0805.4192, eq. 64),
 * and the horizon shell from the current horizon.
 */
inline void BH_XCTS_base::write_regridded_solution(
    base_config_t& target) const {
    target.set(RMID) =
        Kadath::bco_utils::get_radius(space->get_domain(1), OUTER_BC);

    double const conf_inner =
        Kadath::bco_utils::get_boundary_val(2, *conformal_factor, INNER_BC);
    double const est_r_div2 = target(MCH) / (conf_inner * conf_inner);
    target.set(RIN) = est_r_div2;
    target.set(ROUT) = 8. * est_r_div2;

    // the space follows the radii above instead of the default estimates
    target.control(USE_CONFIG_VARS) = true;
    setup_3d_BH_xcts(target);
    target.control(USE_CONFIG_VARS) = false;

    auto const new_spacefile = target.space_filename();
    FILE* ff2 = fopen(new_spacefile.c_str(), "r");
    if (ff2 == nullptr)
        throw std::runtime_error("BH_XCTS::regrid: cannot open " +
                                 new_spacefile);
    base_space_t new_space(ff2);
    Scalar new_conf(new_space, ff2);
    Scalar new_lapse(new_space, ff2);
    Vector new_shift(new_space, ff2);
    fclose(ff2);

    std::cout << "Resolution of old space: ";
    Kadath::bco_utils::print_constant_space_resolution(*space);
    std::cout << "Resolution of new space: ";
    Kadath::bco_utils::print_constant_space_resolution(new_space);
    std::cout << "\nold bounds:" << std::endl;
    Kadath::bco_utils::print_bounds_from_space(*space);
    std::cout << "New bounds:" << std::endl;
    Kadath::bco_utils::print_bounds_from_space(new_space);
    std::cout << std::endl;

    // extend the fields from the horizon shell into the excised domain, since
    // the interpolation near the new excision surface can otherwise pick up
    // the zero excised values
    Scalar old_conf(*conformal_factor);
    Scalar old_lapse(*lapse);
    Vector old_shift(*shift);
    auto const* old_outer_homothetic =
        dynamic_cast<const Domain_shell_outer_homothetic*>(
            space->get_domain(1));
    Kadath::bco_utils::update_adapted_field(old_conf,
                                            2,
                                            1,
                                            old_outer_homothetic,
                                            OUTER_BC);
    Kadath::bco_utils::update_adapted_field(old_lapse,
                                            2,
                                            1,
                                            old_outer_homothetic,
                                            OUTER_BC);
    for (int i = 1; i <= 3; ++i)
        Kadath::bco_utils::update_adapted_field(old_shift.set(i),
                                                2,
                                                1,
                                                old_outer_homothetic,
                                                OUTER_BC);

    new_conf.import(old_conf);
    new_lapse.import(old_lapse);
    for (int i = 1; i <= 3; ++i)
        new_shift.set(i).import(old_shift.set(i));

    for (int d : excluded_doms) {
        new_lapse.set_domain(d).annule_hard();
        new_conf.set_domain(d).annule_hard();
        for (int i = 1; i <= 3; ++i)
            new_shift.set(i).set_domain(d).annule_hard();
    }
    new_lapse.std_base();
    new_conf.std_base();
    new_shift.std_base();

    Kadath::bco_utils::save_to_file(new_space,
                                    target,
                                    new_conf,
                                    new_lapse,
                                    new_shift);
}

}  // namespace Kadath::FUKA_Solvers
