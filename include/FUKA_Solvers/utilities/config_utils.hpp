#pragma once
#include "Configurator/config_bco.hpp"
#include "Configurator/config_binary.hpp"
#include "bco_utilities.hpp"
#include "simple_calculations.hpp"

namespace Kadath::FUKA_Solvers {

template <class config_t>
inline void initialize_binary_config_basics(config_t& bconfig,
                                            double const M1,
                                            double const M2) {
    using namespace ::Kadath::bco_utils;
    using namespace ::Kadath::FUKA_Config;
    check_dist(bconfig(BIN_PARAMS::DIST), M1, M2);

    // Binary Parameters
    bconfig.set(BIN_PARAMS::REXT) = 2 * bconfig(BIN_PARAMS::DIST);

    bconfig.set(BIN_PARAMS::Q) = bconfig(BCO_PARAMS::MADM, NODES::BCO1) /
                                 bconfig(BCO_PARAMS::MCH, NODES::BCO2);
    bconfig.set(BIN_PARAMS::Q) = (bconfig.set(BIN_PARAMS::Q) > 1.0)
                                     ? 1.0 / bconfig.set(BIN_PARAMS::Q)
                                     : bconfig.set(BIN_PARAMS::Q);

    // classical Newtonian estimate
    bconfig.set(BIN_PARAMS::COM) =
        com_estimate(bconfig(BIN_PARAMS::DIST), M1, M2);
}

template <class config_t>
inline void initialize_binary_inspiral_config(config_t& bconfig,
                                              double const M1,
                                              double const M2) {
    using namespace ::Kadath::bco_utils;
    using namespace ::Kadath::FUKA_Config;
    initialize_binary_config_basics(bconfig, M1, M2);

    // obtain 3PN estimate for the global, orbital omega
    KadathPNOrbitalParams(bconfig, M1, M2);

    // delete ADOT, this can always be recalculated during
    // the eccentricity reduction stage
    bconfig.reset(BIN_PARAMS::ADOT);
}

template <class config_t>
inline void set_MIN_SHELL_DR_binary(config_t& bconfig, double M1, double M2) {
    using namespace ::Kadath::bco_utils;
    using namespace ::Kadath::FUKA_Config;
    if (M1 > M2)
        std::swap(M1, M2);
    bconfig.set(BIN_PARAMS::Q) = M2 / M1;

    const double q = bconfig(BIN_PARAMS::Q);
    bconfig.set(BCO_PARAMS::MIN_SHELL_DR, NODES::BCO1) =
        1.7 * std::pow(2.0, shell_factor / q);
    bconfig.set(BCO_PARAMS::MIN_SHELL_DR, NODES::BCO2) =
        1.7 * std::pow(2.0, shell_factor / q);
}

}  // namespace Kadath::FUKA_Solvers