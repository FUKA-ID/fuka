#include "FUKA_Solvers/utilities/binary/binary_setup.hpp"
#include "FUKA_Solvers/utilities/config/config_utils.hpp"
#include "FUKA_Solvers/utilities/config/simple_calculations.hpp"
#include "FUKA_Solvers/utilities/drivers/solve_BH_for_binary.hpp"
#include "FUKA_Solvers/utilities/drivers/solve_NS_for_binary.hpp"
#include "FUKA_Solvers/utilities/fields/scalar_calculations.hpp"
#include "kadath_adapted_bh.hpp"
/**
 * \addtogroup BNS_XCTS
 * \ingroup FUKA
 * @{*/
#include "EOS/FUKA_EOS_Utilities.hh"
using namespace ::Kadath::FUKA_Config;

namespace Kadath {
namespace FUKA_Solvers {
template <class config_t>
inline void BNS_XCTS_setup_bin_config(config_t& bconfig) {
    setup_binary_config(bconfig,
                        bconfig(BCO_PARAMS::MADM, NODES::BCO1),
                        bconfig(BCO_PARAMS::MADM, NODES::BCO2));
}

template <class config_t>
void BNS_XCTS_setup_space(config_t& bconfig) {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::array<std::string, 2> filenames;

    filenames[0] = solve_NS_for_binary(bconfig, NODES::BCO1);
    filenames[1] = solve_NS_for_binary(bconfig, NODES::BCO2);

    // debugging only
    for (auto& f : filenames)
        if (rank == 0)
            std::cout << f << std::endl;

    if (rank == 0)
        BNS_XCTS_superimposed_import(bconfig, filenames);
    MPI_Barrier(MPI_COMM_WORLD);
    bconfig.control(CONTROLS::SEQUENCES) = false;
    bconfig.open_config();
}

template <typename eos_t>
struct BNS_XCTS_setup_boosted_3d {
    inline void operator()(kadath_config_boost<BCO_NS_INFO>& NS1config,
                           kadath_config_boost<BCO_NS_INFO>& NS2config,
                           kadath_config_boost<BIN_INFO>& bconfig) {
        using namespace ::Kadath::bco_utils;
        using namespace binary_setup;
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        Isolated_NS<eos_t> ns1(NS1config);
        Isolated_NS<eos_t> ns2(NS2config);

        // the binary starts from the isolated stars' parameters
        for (auto i : BCO_PARAMS_ARY) {
            bconfig.set(i, NODES::BCO1) = NS1config.set(i);
            bconfig.set(i, NODES::BCO2) = NS2config.set(i);
        }

        set_companion_outer_radii(bconfig);
        set_MIN_SHELL_DR_binary(bconfig,
                                bconfig(BCO_PARAMS::MADM, NODES::BCO1),
                                bconfig(BCO_PARAMS::MADM, NODES::BCO2));

        std::vector<double> const out_bounds{outer_shell_bounds(bconfig)};
        std::vector<double> const NS1_bounds{companion_bounds(bconfig,
                                                              ns1.space(),
                                                              *ns1.sol.conf,
                                                              *ns1.sol.shift,
                                                              {0, 1},
                                                              2,
                                                              NODES::BCO1)};
        std::vector<double> const NS2_bounds{companion_bounds(bconfig,
                                                              ns2.space(),
                                                              *ns2.sol.conf,
                                                              *ns2.sol.shift,
                                                              {0, 1},
                                                              2,
                                                              NODES::BCO2)};
        if (rank == 0) {
            std::cout << "Bounds:" << std::endl;
            print_bounds("NS1-bounds", NS1_bounds);
            print_bounds("NS2-bounds", NS2_bounds);
            print_bounds("Outer", out_bounds);
        }

        Space_bin_ns space(CHEB_TYPE,
                           bconfig(BIN_PARAMS::DIST),
                           NS1_bounds,
                           NS2_bounds,
                           out_bounds,
                           bconfig(BIN_PARAMS::BIN_RES),
                           bconfig(BCO_PARAMS::NINSHELLS, NODES::BCO1),
                           bconfig(BCO_PARAMS::NINSHELLS, NODES::BCO2));
        Base_tensor basis(space, CARTESIAN_BASIS);

        ns1.map_surface(space, space.ADAPTED1);
        ns2.map_surface(space, space.ADAPTED2);

        double const xc1 = get_center(space, space.NS1);
        double const xc2 = get_center(space, space.NS2);
        if (rank == 0) {
            std::cout << "xc1: " << xc1 << std::endl;
            std::cout << "xc2: " << xc2 << std::endl;
        }

        double const ns1_invw4 = set_decay(bconfig, NODES::BCO1);
        double const ns2_invw4 = set_decay(bconfig, NODES::BCO2);
        if (rank == 0)
            std::cout << "WeightNS1: "
                      << bconfig(BCO_PARAMS::DECAY, NODES::BCO1) << ", "
                      << "WeightNS2: "
                      << bconfig(BCO_PARAMS::DECAY, NODES::BCO2) << std::endl;

        Superposed_fields
            f(space, basis, ns1.sol, ns2.sol, xc1, xc2, ns1_invw4, ns2_invw4);

        // no matter and no velocity potential outside the stars
        int const ndom = space.get_nbr_domains();
        for (int i = space.ADAPTED1 + 1; i < space.NS2; ++i) {
            f.phi.set_domain(i).annule_hard();
            f.logh.set_domain(i).annule_hard();
        }
        for (int i = space.ADAPTED2 + 1; i < ndom; ++i) {
            f.phi.set_domain(i).annule_hard();
            f.logh.set_domain(i).annule_hard();
        }
        f.std_base();

        // the preconditioning stage starts an inspiral from the initial guess
        bconfig.set_stage(STAGES::PRE) = !bconfig.set_stage(STAGES::HEADON);

        save_to_file(space, bconfig, f.conf, f.lapse, f.shift, f.logh, f.phi);
    }
};

template <class config_t>
void BNS_XCTS_superimposed_import(config_t& bconfig,
                                  std::array<std::string, 2> NSfilenames) {
    using namespace Kadath::FUKA_EOS;

    // load single NS configuration
    std::string ns1filename{NSfilenames[0]};
    kadath_config_boost<BCO_NS_INFO> NS1config(ns1filename);

    std::string ns2filename{NSfilenames[1]};
    kadath_config_boost<BCO_NS_INFO> NS2config(ns2filename);

    // setup eos and update central density
    const std::string eos_type =
        NS1config.eos<std::string>(EOS_PARAMS::EOSTYPE);
    EOS_Function_Dispatcher::dispatch<BNS_XCTS_setup_boosted_3d>(bconfig,
                                                                 eos_type,
                                                                 NS1config,
                                                                 NS2config,
                                                                 bconfig);
}

/** @}*/
}  // namespace FUKA_Solvers
}  // namespace Kadath
