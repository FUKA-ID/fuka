#pragma once
#include "FUKA_Solvers/NS_XCTS/utilities/boost/Boosted_NS_XCTS.hpp"
#include "FUKA_Solvers/utilities/setup_xcts_from_iso.hpp"
#include "NS_XCTS.hpp"
#include "name_tools.hpp"

/**
 * \addtogroup NS_XCTS
 * \ingroup FUKA
 * @{*/
namespace Kadath::FUKA_Solvers {

inline int NS_XCTS_driver(NS_XCTS_BASE::base_config_t& bconfig,
                          ns_sequence const& seq,
                          Parameter_sequence<BCO_PARAMS>& resolution,
                          std::string outputdir);

inline NS_XCTS_BASE::base_config_t NS_XCTS_sequence_setup(
    NS_XCTS_BASE::base_config_t& seqconfig,
    std::string outputdir);

inline int NS_XCTS_solver_driver(NS_XCTS_BASE::base_config_t& bconfig,
                                 ns_sequence const& seq,
                                 Parameter_sequence<BCO_PARAMS>& resolution,
                                 std::string outputdir);

int NS_XCTS_seq_driver(NS_XCTS_BASE::base_config_t& seqconfig,
                       ns_sequence const& seq,
                       Parameter_sequence<BCO_PARAMS>& resolution,
                       std::string const outputdir);

inline int launch_final_stage_driver(NS_XCTS_BASE::base_config_t& bconfig,
                                     ns_sequence const& seq,
                                     Parameter_sequence<BCO_PARAMS>& resolution,
                                     std::string outputdir);

template <typename config_t, class Res_t>
inline int NS_XCTS_binary_boost_driver(config_t& bconfig,
                                       Res_t& resolution,
                                       std::string outputdir,
                                       kadath_config_boost<BIN_INFO> binconfig,
                                       const size_t bco);

/** @}*/
};  // namespace Kadath::FUKA_Solvers

#include "NS_XCTS_driver.cpp"