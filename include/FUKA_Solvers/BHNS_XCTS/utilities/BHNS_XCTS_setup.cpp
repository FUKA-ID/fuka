#include "EOS/FUKA_EOS_Utilities.hh"
#include "FUKA_Solvers/utilities/binary/binary_setup.hpp"
#include "FUKA_Solvers/utilities/config/config_utils.hpp"
#include "FUKA_Solvers/utilities/drivers/solve_BH_for_binary.hpp"
#include "FUKA_Solvers/utilities/drivers/solve_NS_for_binary.hpp"
#include "kadath_bhns.hpp"

namespace Kadath::FUKA_Solvers {

template <class config_t>
void BHNS_XCTS_setup_bin_config(config_t& bconfig) {
    setup_binary_config(bconfig,
                        bconfig(BCO_PARAMS::MADM, NODES::BCO1),
                        bconfig(BCO_PARAMS::MCH, NODES::BCO2));
}

template <class config_t>
void BHNS_XCTS_setup_space(config_t& bconfig) {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    std::array<std::string, 2> const filenames{
        solve_NS_for_binary(bconfig, NODES::BCO1),
        solve_BH_for_binary(bconfig, NODES::BCO2)};
    if (rank == 0) {
        for (auto const& f : filenames)
            std::cout << f << std::endl;
        BHNS_XCTS_superimposed_import(bconfig, filenames);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    bconfig.control(CONTROLS::SEQUENCES) = false;
    bconfig.open_config();
}

/// Initial guess of a BHNS binary from the isolated star and hole.
template <typename eos_t>
struct BHNS_XCTS_setup_boosted_3d {
    void operator()(kadath_config_boost<BCO_NS_INFO>& NSconfig,
                    kadath_config_boost<BCO_BH_INFO>& BHconfig,
                    kadath_config_boost<BIN_INFO>& bconfig) {
        using namespace ::Kadath::bco_utils;
        using namespace binary_setup;
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        Isolated_NS<eos_t> ns(NSconfig);
        Isolated_BH bh(BHconfig);

        set_companion_outer_radii(bconfig);
        set_min_shell_dr(bconfig, bconfig.set(BIN_PARAMS::Q));

        std::vector<double> const out_bounds{outer_shell_bounds(bconfig)};
        std::vector<double> const NS_bounds{companion_bounds(bconfig,
                                                             ns.space(),
                                                             *ns.sol.conf,
                                                             *ns.sol.shift,
                                                             {0, 1},
                                                             2,
                                                             NODES::BCO1)};
        std::vector<double> const BH_bounds{companion_bounds(bconfig,
                                                             bh.space(),
                                                             *bh.sol.conf,
                                                             *bh.sol.shift,
                                                             {0, 1},
                                                             2,
                                                             NODES::BCO2)};
        if (rank == 0) {
            std::cout << "Local bounds:" << std::endl;
            print_bounds("NS-bounds", NS_bounds);
            print_bounds("BH-bounds", BH_bounds);
            print_bounds("outer-bounds", out_bounds);
        }

        Space_bhns space(CHEB_TYPE,
                         bconfig(BIN_PARAMS::DIST),
                         NS_bounds,
                         BH_bounds,
                         out_bounds,
                         bconfig(BIN_PARAMS::BIN_RES),
                         bconfig(BCO_PARAMS::NINSHELLS, NODES::BCO1));
        Base_tensor basis(space, CARTESIAN_BASIS);

        bh.extend_into_excision();
        ns.map_surface(space, space.ADAPTEDNS);

        double const xc1 = get_center(space, space.NS);
        double const xc2 = get_center(space, space.BH);
        if (rank == 0) {
            std::cout << "xc1: " << xc1 << std::endl;
            std::cout << "xc2: " << xc2 << std::endl;
        }

        double const ns_invw4 = set_decay(bconfig, NODES::BCO1);
        double const bh_invw4 = set_decay(bconfig, NODES::BCO2);
        if (rank == 0)
            std::cout << "WeightNS: " << bconfig(BCO_PARAMS::DECAY, NODES::BCO1)
                      << ", "
                      << "WeightBH: " << bconfig(BCO_PARAMS::DECAY, NODES::BCO2)
                      << std::endl;

        Superposed_fields
            f(space, basis, ns.sol, bh.sol, xc1, xc2, ns_invw4, bh_invw4);

        // no matter around the hole, nothing inside the horizon
        for (int d = space.BH; d < space.OUTER; ++d) {
            f.logh.set_domain(d).annule_hard();
            f.phi.set_domain(d).annule_hard();
        }
        for (int d = space.BH; d < space.BH + 2; ++d) {
            f.conf.set_domain(d).annule_hard();
            f.lapse.set_domain(d).annule_hard();
        }
        f.std_base();

        save_to_file(space, bconfig, f.conf, f.lapse, f.shift, f.logh, f.phi);
    }
};

template <class config_t>
void BHNS_XCTS_superimposed_import(config_t& bconfig,
                                   std::array<std::string, 2> co_filenames) {
    using namespace Kadath::FUKA_EOS;
    kadath_config_boost<BCO_NS_INFO> NSconfig(co_filenames[0]);
    kadath_config_boost<BCO_BH_INFO> BHconfig(co_filenames[1]);

    const std::string eos_type = NSconfig.eos<std::string>(EOS_PARAMS::EOSTYPE);
    EOS_Function_Dispatcher::dispatch<BHNS_XCTS_setup_boosted_3d>(bconfig,
                                                                  eos_type,
                                                                  NSconfig,
                                                                  BHconfig,
                                                                  bconfig);
}

}  // namespace Kadath::FUKA_Solvers
