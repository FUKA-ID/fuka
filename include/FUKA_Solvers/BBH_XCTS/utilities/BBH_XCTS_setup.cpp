#include "FUKA_Solvers/utilities/binary/binary_setup.hpp"
#include "FUKA_Solvers/utilities/config/config_utils.hpp"
#include "FUKA_Solvers/utilities/drivers/solve_BH_for_binary.hpp"
#include "kadath_bin_bh.hpp"

namespace Kadath::FUKA_Solvers {

template <class config_t>
void BBH_XCTS_setup_bin_config(config_t& bconfig) {
    setup_binary_config(bconfig,
                        bconfig(BCO_PARAMS::MCH, NODES::BCO1),
                        bconfig(BCO_PARAMS::MCH, NODES::BCO2));
}

template <class config_t>
void BBH_XCTS_setup_space(config_t& bconfig) {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    std::array<std::string, 2> const filenames{
        solve_BH_for_binary(bconfig, NODES::BCO1),
        solve_BH_for_binary(bconfig, NODES::BCO2)};
    if (rank == 0)
        BBH_XCTS_superimposed_import(bconfig, filenames);
    MPI_Barrier(MPI_COMM_WORLD);
    bconfig.open_config();
    bconfig.control(CONTROLS::SEQUENCES) = false;
}

inline void BBH_XCTS_superimposed_import(
    kadath_config_boost<BIN_INFO>& bconfig,
    std::array<std::string, 2> const& bh_filenames) {
    using namespace ::Kadath::bco_utils;
    using namespace binary_setup;

    kadath_config_boost<BCO_BH_INFO> BH1config(bh_filenames[0]);
    kadath_config_boost<BCO_BH_INFO> BH2config(bh_filenames[1]);
    Isolated_BH bh1(BH1config);
    Isolated_BH bh2(BH2config);

    // the binary starts from the isolated holes' parameters, but keeps its
    // shells
    int const nshells1 = bconfig(BCO_PARAMS::NSHELLS, NODES::BCO1);
    int const nshells2 = bconfig(BCO_PARAMS::NSHELLS, NODES::BCO2);
    for (auto idx : BCO_PARAMS_ARY) {
        bconfig.set(idx, NODES::BCO1) = BH1config.set(idx);
        bconfig.set(idx, NODES::BCO2) = BH2config.set(idx);
    }
    bconfig.set(BCO_PARAMS::NSHELLS, NODES::BCO1) = nshells1;
    bconfig.set(BCO_PARAMS::NSHELLS, NODES::BCO2) = nshells2;

    set_companion_outer_radii(bconfig);
    set_min_shell_dr(bconfig, bconfig.set(BIN_PARAMS::Q));

    std::vector<double> const BH1_bounds{companion_bounds(bconfig,
                                                          bh1.space(),
                                                          *bh1.sol.conf,
                                                          *bh1.sol.shift,
                                                          {0, 1},
                                                          2,
                                                          NODES::BCO1)};
    std::vector<double> const BH2_bounds{companion_bounds(bconfig,
                                                          bh2.space(),
                                                          *bh2.sol.conf,
                                                          *bh2.sol.shift,
                                                          {0, 1},
                                                          2,
                                                          NODES::BCO2)};
    std::vector<double> const out_bounds{outer_shell_bounds(bconfig)};

    Space_bin_bh space(CHEB_TYPE,
                       bconfig(BIN_PARAMS::DIST),
                       BH1_bounds,
                       BH2_bounds,
                       out_bounds,
                       bconfig(BIN_PARAMS::BIN_RES));
    Base_tensor basis(space, CARTESIAN_BASIS);

    bh1.extend_into_excision(true);
    bh2.extend_into_excision(true);

    double const xc1 = get_center(space, space.BH1);
    double const xc2 = get_center(space, space.BH2);
    Superposed_fields f(space,
                        basis,
                        bh1.sol,
                        bh2.sol,
                        xc1,
                        xc2,
                        set_decay(bconfig, NODES::BCO1),
                        set_decay(bconfig, NODES::BCO2));

    // nothing inside the horizons
    for (int const nuc : {space.BH1, space.BH2})
        for (int const d : {nuc, nuc + 1}) {
            f.conf.set_domain(d).annule_hard();
            f.lapse.set_domain(d).annule_hard();
            for (int i = 1; i <= 3; i++)
                f.shift.set(i).set_domain(d).annule_hard();
        }
    f.conf.std_base();
    f.lapse.std_base();
    f.shift.std_base();

    save_to_file(space, bconfig, f.conf, f.lapse, f.shift);
}

}  // namespace Kadath::FUKA_Solvers
