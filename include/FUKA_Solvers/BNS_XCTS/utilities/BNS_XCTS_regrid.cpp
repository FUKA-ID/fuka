#include <string>
#include <vector>
#include "FUKA_Solvers/utilities/config_utils.hpp"
#include "FUKA_Solvers/utilities/scalar_calculations.hpp"
#include "Solvers/fuka_syst/fuka_syst_tools.hpp"
#include "kadath_bin_ns.hpp"

namespace Kadath::FUKA_Solvers {
bool BNS_XCTS_base::increment_resolution() {
    // Currently TOTAL_BC is the only stage that computes solutions in hydrostratic equilibrium
    if (!(resolution->final() > resolution->init()) ||
        solver_stage != STAGES::TOTAL_BC)
        return false;

    auto resolution_indices = resolution->get_indices();
    auto const& final_res = resolution->final();
    if ((*bconfig)(resolution_indices) >= final_res) {
        return false;
    }

    int next_res =
        Kadath::bco_utils::next_resolution((*bconfig)(resolution_indices));

    // FUKA is expensive, so we do not exceed the desired resolution
    if (next_res <= final_res) {
        bconfig->set(resolution_indices) = next_res;
    } else {
        return false;
    }
    return true;
}

inline void BNS_XCTS_base::regrid() {
    std::string output_fname = "bns_regrid.info";

    if (rank == 0) {
        auto M_1 = (*bconfig)(BCO_PARAMS::MADM, NODES::BCO1);
        auto M_2 = (*bconfig)(BCO_PARAMS::MADM, NODES::BCO2);
        set_MIN_SHELL_DR_binary(*bconfig, M_1, M_2);

        if (std::isnan(bconfig->set(BIN_PARAMS::OUTER_SHELLS)))
            bconfig->set(BIN_PARAMS::OUTER_SHELLS) = 0;

        const std::array<int, 2> old_adapted_doms{space->ADAPTED1,
                                                  space->ADAPTED2};

        std::array<const Domain_shell_outer_adapted*, 2> old_outer_adapted;
        std::array<const Domain_shell_inner_adapted*, 2> old_inner_adapted;

        for (int i = 0; i < 2; ++i) {
            int const d = old_adapted_doms[i];
            old_outer_adapted[i] =
                dynamic_cast<const Domain_shell_outer_adapted*>(
                    space->get_domain(d));
            old_inner_adapted[i] =
                dynamic_cast<const Domain_shell_inner_adapted*>(
                    space->get_domain(d + 1));
        }

        int ndim = 3;
        int type_coloc = space->get_type_base();

        Kadath::bco_utils::update_config_NS_radii(*space,
                                                  *bconfig,
                                                  space->ADAPTED1,
                                                  NODES::BCO1);

        Kadath::bco_utils::update_config_NS_radii(*space,
                                                  *bconfig,
                                                  space->ADAPTED2,
                                                  NODES::BCO1);

        double rmax = std::max((*bconfig)(BCO_PARAMS::RMID, NODES::BCO1),
                               (*bconfig)(BCO_PARAMS::RMID, NODES::BCO2));

        const double rout_sep_est =
            ((*bconfig)(BIN_PARAMS::DIST) / 2. - rmax) / 3. + rmax;
        const double rout_max_est = Kadath::bco_utils::gold_ratio * rmax;
        bconfig->set(BCO_PARAMS::ROUT, NODES::BCO1) = rout_sep_est;
        bconfig->set(BCO_PARAMS::ROUT, NODES::BCO2) =
            (*bconfig)(BCO_PARAMS::ROUT, NODES::BCO1);

        // create old radius scalar field
        Scalar old_space_radius(*space);
        old_space_radius.annule_hard();

        for (int d = 0; d < ndom; ++d) {
            old_space_radius.set_domain(d) = space->get_domain(d)->get_radius();
        }

        // Get adapted domain radii
        for (int i = 0; i < 2; ++i) {
            int const dom = old_adapted_doms[i];
            old_space_radius.set_domain(dom) =
                old_outer_adapted[i]->get_outer_radius();
        }
        old_space_radius.std_base();
        // end creating old radius scalar fields

        // setup bounds for creating new space
        std::vector<double> out_bounds(1 +
                                       (*bconfig)(BIN_PARAMS::OUTER_SHELLS));
        for (int e = 0; e < out_bounds.size(); ++e)
            out_bounds[e] = (*bconfig)(BIN_PARAMS::REXT) * (1. + e * 0.25);

        std::vector<int> ns_interior_doms{
            FUKA_Syst_tools::vector_of_domains(space->NS1, space->ADAPTED1)};
        std::vector<int> exclusion_doms{
            FUKA_Syst_tools::vector_of_domains(space->NS2, space->ADAPTED2)};
        // concat domain lists together
        std::for_each(ns_interior_doms.rbegin(),
                      ns_interior_doms.rend(),
                      [&exclusion_doms](auto e) {
                          auto it = exclusion_doms.begin();
                          exclusion_doms.insert(it, e);
                      });
        auto drPsi(compute_drPsi(*space,
                                 *conformal_factor,
                                 Metric_flat(*space, shift->get_basis()),
                                 exclusion_doms,
                                 space->OUTER));
        std::vector<double> NS1_bounds{
            Kadath::bco_utils::set_arb_boundsv3(*bconfig,
                                                drPsi,
                                                space->ADAPTED1 + 1,
                                                NODES::BCO1)};

        std::vector<double> NS2_bounds{
            Kadath::bco_utils::set_arb_boundsv3(*bconfig,
                                                drPsi,
                                                space->ADAPTED2 + 1,
                                                NODES::BCO1)};
        // end setup bounds

        // print bounds to stdout - debugging only
        std::cout << "Local bounds:" << std::endl;
        Kadath::bco_utils::print_bounds("NS1-bounds", NS1_bounds);
        Kadath::bco_utils::print_bounds("NS2-bounds", NS2_bounds);
        Kadath::bco_utils::print_bounds("outer-bounds", out_bounds);
        std::cout << std::endl;

        Space_bin_ns new_space(type_coloc,
                               (*bconfig)(BIN_PARAMS::DIST),
                               NS1_bounds,
                               NS2_bounds,
                               out_bounds,
                               (*bconfig)(BIN_PARAMS::BIN_RES),
                               (*bconfig)(BCO_PARAMS::NINSHELLS, NODES::BCO1),
                               (*bconfig)(BCO_PARAMS::NINSHELLS, NODES::BCO2));
        Base_tensor new_basis(new_space, CARTESIAN_BASIS);

        std::cout << "Resolution of old space: ";
        Kadath::bco_utils::print_constant_space_resolution(*space);

        std::cout << "Resolution of new space: ";
        Kadath::bco_utils::print_constant_space_resolution(new_space);

        std::cout << "\nOld bounds:" << std::endl;
        Kadath::bco_utils::print_bounds_from_space(*space);

        std::cout << "New bounds:" << std::endl;
        Kadath::bco_utils::print_bounds_from_space(new_space);

        const std::array<int, 2> new_adapted_doms{space->ADAPTED1,
                                                  space->ADAPTED2};
        const std::array<int, 2> new_nuc_doms{space->NS1, space->NS2};
        const std::array<double, 2> xc{
            space->get_domain(space->NS1)->get_center()(1),
            space->get_domain(space->NS2)->get_center()(1)};

        std::array<const Domain_shell_outer_adapted*, 2> new_outer_adapted;
        std::array<const Domain_shell_inner_adapted*, 2> new_inner_adapted;

        for (int i = 0; i < 2; ++i) {
            auto& d = new_adapted_doms[i];
            new_outer_adapted[i] =
                dynamic_cast<const Domain_shell_outer_adapted*>(
                    space->get_domain(d));
            new_inner_adapted[i] =
                dynamic_cast<const Domain_shell_inner_adapted*>(
                    space->get_domain(d + 1));
        }

        for (int i = 0; i < 2; ++i) {
            int const dom = old_adapted_doms[i];

            // Updated mapping for NS
            Kadath::bco_utils::interp_adapted_mapping(new_inner_adapted[i],
                                                      dom,
                                                      old_space_radius);
            Kadath::bco_utils::interp_adapted_mapping(new_outer_adapted[i],
                                                      dom,
                                                      old_space_radius);

            // Interpolate old_phi field outside of the star for import
            Kadath::bco_utils::update_adapted_field(*phi,
                                                    dom,
                                                    dom + 1,
                                                    old_inner_adapted[i],
                                                    INNER_BC);
        }

        // setup new fields
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
        // end setup

        // import old fields into new space
        new_conf.import(*conformal_factor);
        new_lapse.import(*lapse);
        new_logh.import(*logh);
        new_phi.import(*phi);

        new_shift.set(1).import(shift->set(1));
        new_shift.set(2).import(shift->set(2));
        new_shift.set(3).import(shift->set(3));
        // end import

        // make sure there is no matter or vel.pot. outside of the star
        for (auto& dom : new_adapted_doms) {
            new_logh.set_domain(dom + 1).annule_hard();
            new_phi.set_domain(dom + 1).annule_hard();
        }
        for (int dom = space->OUTER; dom < ndom; ++dom) {
            new_logh.set_domain(dom + 1).annule_hard();
            new_phi.set_domain(dom + 1).annule_hard();
        }

        new_lapse.std_base();
        new_conf.std_base();
        new_logh.std_base();
        new_shift.std_base();
        new_phi.std_base();

        bconfig->set_filename(output_fname);
        Kadath::bco_utils::save_to_file(new_space,
                                        *bconfig,
                                        new_conf,
                                        new_lapse,
                                        new_shift,
                                        new_logh,
                                        new_phi);
    }
}
}  // namespace Kadath::FUKA_Solvers