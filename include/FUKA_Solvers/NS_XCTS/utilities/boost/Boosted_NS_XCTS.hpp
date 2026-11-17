/**
 * @file Boosted_NS_XCTS.hpp
 * @author Samuel Tootle (sdtootle@gmail.com)
 * @brief This module takes an isolated NS solution solver and
 * modifies it to produce a boosted NS solution suitable for
 * use in binary initial data computations.
 * branch
 * @date 2024-02-21
 *
 * @copyright Copyright (c) 2024, GNU General Public Licensev3
 * This file is part of the KADATH library and published under
 * https://arxiv.org/abs/2103.09911
 */
#pragma once
#include "../../NS_XCTS.hpp"
#include "codes_utilities.hpp"

/**
 * \addtogroup NS_XCTS
 * \ingroup FUKA
 * @{*/
namespace Kadath::FUKA_Solvers {

template <class eos_t>
struct Boosted_NS_XCTS : public NS_XCTS_UNIFORM_ROT<eos_t> {
    using binary_config_t =
        Kadath::FUKA_Config::kadath_config_boost<Kadath::FUKA_Config::BIN_INFO>;

    using typename NS_XCTS_UNIFORM_ROT<eos_t>::base_config_t;
    using typename NS_XCTS_UNIFORM_ROT<eos_t>::base_space_t;
    using NS_XCTS_UNIFORM_ROT<eos_t>::syst;
    using NS_XCTS_UNIFORM_ROT<eos_t>::bconfig;
    using NS_XCTS_UNIFORM_ROT<eos_t>::syst_init;
    using NS_XCTS_UNIFORM_ROT<eos_t>::seq;
    using NS_XCTS_UNIFORM_ROT<eos_t>::ndom;
    using NS_XCTS_UNIFORM_ROT<eos_t>::space;
    using NS_XCTS_UNIFORM_ROT<eos_t>::stagename;
    using NS_XCTS_UNIFORM_ROT<eos_t>::coord_vectors;
    using NS_XCTS_UNIFORM_ROT<eos_t>::cfields;
    using NS_XCTS_UNIFORM_ROT<eos_t>::rank;
    using NS_XCTS_UNIFORM_ROT<eos_t>::reload;

    ptr_data_member(Kadath::Scalar, phi, unique);

    internal_variable(double, H_scale);

   protected:
    void load_solution_from_file() override;
    void save_solution_to_file() const override;

   public:
    /// Members a revert needs (Solution_snapshot).
    static constexpr auto snapshot_members() {
        return std::tuple_cat(NS_XCTS_BASE::snapshot_members(),
                              std::make_tuple(&Boosted_NS_XCTS::phi));
    }

    Boosted_NS_XCTS(base_config_t* config_,
                    ns_sequence const& seq_,
                    Parameter_sequence<BCO_PARAMS> const& res_,
                    std::string outputdir_,
                    binary_config_t* binary_config_ = nullptr,
                    int const rank_ = 0);

    void setup_syst() override;
    void reset_all_ptrs() override;

   private:
    /// Non-owning: the binary config (DIST, GOMEGA) of the caller.
    binary_config_t* binary_config;
};
}  // namespace Kadath::FUKA_Solvers

#include "Boosted_NS_XCTS_imp.cpp"
