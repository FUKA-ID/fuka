/**
 * @file FUKA_XCTS_Binary_base.hpp
 * @brief State and I/O shared by the binary XCTS solvers: the metric part
 * (BBH, BNS, BHNS) and the matter part (BNS, BHNS).
 *
 * @copyright Copyright (c) 2026, GNU General Public Licensev3
 * This file is part of the KADATH library and published under
 * https://arxiv.org/abs/2103.09911
 */
#pragma once
#include <array>
#include <iomanip>
#include <memory>
#include <sstream>
#include <string>
#include <tuple>

#include "Configurator/config_binary.hpp"
#include "FUKA_Solvers/FUKA_XCTS_Solver_base.hpp"
#include "Solvers/sequences/bco_sequence.hpp"
#include "Solvers/sequences/parameter_sequence.hpp"

/** \addtogroup Solver_base
 * \ingroup FUKA
 * @{
 */
namespace Kadath::FUKA_Solvers {

/**
 * @brief Binary solution: conformal factor, lapse and shift on a binary
 * space, the two companions' sequences, the resolution sequence, and the
 * binary coordinate-field update. The solution file starts with (space, P,
 * N, beta); solvers with matter append their fields (load_matter_fields).
 *
 * @tparam space_t binary space (Space_bin_bh, Space_bin_ns, Space_bhns)
 */
template <class space_t>
struct FUKA_XCTS_Binary_base
    : public FUKA_XCTS_Solver_base<space_t, kadath_config_boost<BIN_INFO>> {
    using solver_base_t =
        FUKA_XCTS_Solver_base<space_t, kadath_config_boost<BIN_INFO>>;
    using typename solver_base_t::base_config_t;
    using typename solver_base_t::base_space_t;
    using typename solver_base_t::cfary_t;
    using typename solver_base_t::cfgen_t;
    using bco_seq_t = bco_sequence<BCO_PARAMS, NODES>;
    using bin_res_seq_t = Parameter_sequence<BIN_PARAMS>;

   protected:
    using solver_base_t::basis;
    using solver_base_t::bconfig;
    using solver_base_t::cfields;
    using solver_base_t::coord_vectors;
    using solver_base_t::fmet;
    using solver_base_t::ndom;
    using solver_base_t::outputdir;
    using solver_base_t::rank;
    using solver_base_t::space;
    using solver_base_t::syst;

    // Sequences of the two companions and of the resolution
    ptr_data_member(bco_seq_t const, bco1_seq, unique);
    ptr_data_member(bco_seq_t const, bco2_seq, unique);
    ptr_data_member(bin_res_seq_t, resolution, unique);

    // Solution
    ptr_data_member(Scalar, conformal_factor, unique);
    ptr_data_member(Scalar, lapse, unique);
    ptr_data_member(Vector, shift, unique);

    // x-coordinates of the companions' centers and of the outer domains
    internal_variable(double, xc1);
    internal_variable(double, xc2);
    internal_variable(double, xo);

   protected:
    FUKA_XCTS_Binary_base(base_config_t* config_,
                          bco_seq_t const& bco1_seq_,
                          bco_seq_t const& bco2_seq_,
                          bin_res_seq_t const& res_,
                          std::string const& outputdir_,
                          int const rank_)
        : solver_base_t(config_, outputdir_, rank_),
          bco1_seq(std::make_unique<bco_seq_t const>(bco1_seq_)),
          bco2_seq(std::make_unique<bco_seq_t const>(bco2_seq_)),
          resolution(std::make_unique<bin_res_seq_t>(res_)),
          xc1(0),
          xc2(0),
          xo(0) {
        if (rank == 0) {
            std::cout << *bco1_seq << std::endl
                      << *bco2_seq << std::endl
                      << *resolution << std::endl;
        }
    }

    /// Nucleus domains of the first and second companion.
    virtual std::array<int, 2> nucleus_domains() const = 0;

    /// Read the space from an open solution file.
    virtual std::unique_ptr<base_space_t> read_space(FILE* ff) const {
        return std::make_unique<base_space_t>(ff);
    }

    void initialize_support_containers() override {
        basis = std::make_unique<Base_tensor>(shift->get_basis());
        fmet = std::make_unique<Metric_flat>(*space, *basis);
        cfields = std::make_unique<cfgen_t>(*space);
        coord_vectors =
            std::make_unique<cfary_t>(default_binary_vector_ary(*space));

        auto const [nuc1, nuc2] = nucleus_domains();
        xc1 = bco_utils::get_center(*space, nuc1);
        xc2 = bco_utils::get_center(*space, nuc2);
        xo = bco_utils::get_center(*space, ndom - 1);
    }

    void save_solution_to_file() const override {
        bconfig->set_outputdir(outputdir);
        Kadath::bco_utils::save_to_file(*space,
                                        *bconfig,
                                        *conformal_factor,
                                        *lapse,
                                        *shift);
    }

    /// Read the fields stored after (P, N, beta).
    virtual void load_matter_fields(FILE* /*ff*/) {}

    void load_solution_from_file() override {
        std::string const spacein{bconfig->space_filename()};
        FILE* ff1 = fopen(spacein.c_str(), "r");
        if (ff1 == nullptr)
            throw std::runtime_error("cannot open " + spacein);

        space = read_space(ff1);
        conformal_factor = std::make_unique<Scalar>(*space, ff1);
        lapse = std::make_unique<Scalar>(*space, ff1);
        shift = std::make_unique<Vector>(*space, ff1);
        load_matter_fields(ff1);
        fclose(ff1);

        conformal_factor->coef();
        lapse->coef();
        shift->coef();

        ndom = space->get_nbr_domains();
    }

    void update_coordinate_fields() override {
        Kadath::update_fields(*cfields,
                              *coord_vectors,
                              {},
                              xo,
                              xc1,
                              xc2,
                              syst.get());
    }

    bool is_compatible_solution(base_config_t& old_config) override {
        return bconfig->set(BCO_PARAMS::NSHELLS, NODES::BCO1) ==
                   old_config.set(BCO_PARAMS::NSHELLS, NODES::BCO1) &&
               bconfig->set(BCO_PARAMS::NSHELLS, NODES::BCO2) ==
                   old_config.set(BCO_PARAMS::NSHELLS, NODES::BCO2);
    }

    /**
     * @brief Converged filename `<prefix>[_<stage>].[<eos>.]<dist>.<chi1>.
     * <chi2>.<Mtot>.q<q>.<nshells1>.<nshells2>.<res>`; sets Q in the config.
     *
     * @param prefix solver prefix (BBH, BNS, BHNS)
     * @param stage stage name
     * @param M1 mass of the first companion
     * @param M2 mass of the second companion
     * @param eosname EOS name, omitted if empty
     */
    std::string binary_converged_filename(std::string const& prefix,
                                          std::string const& stage,
                                          double M1,
                                          double M2,
                                          std::string const& eosname) const {
        auto const res = space->get_domain(0)->get_nbr_points()(0);
        if (M2 > M1)
            std::swap(M1, M2);
        bconfig->set(BIN_PARAMS::Q) = M2 / M1;
        auto& cfg = *bconfig;

        std::stringstream ss;
        ss << prefix;
        if (stage != "")
            ss << "_" << stage << ".";
        else
            ss << ".";
        if (!eosname.empty())
            ss << eosname << ".";
        ss << cfg(BIN_PARAMS::DIST) << "." << cfg(BCO_PARAMS::CHI, NODES::BCO1)
           << "." << cfg(BCO_PARAMS::CHI, NODES::BCO2) << "." << M1 + M2 << ".q"
           << cfg(BIN_PARAMS::Q) << "." << std::setfill('0') << std::setw(1)
           << cfg(BCO_PARAMS::NSHELLS, NODES::BCO1) << "." << std::setfill('0')
           << std::setw(1) << cfg(BCO_PARAMS::NSHELLS, NODES::BCO2) << "."
           << std::setfill('0') << std::setw(2) << res;
        return ss.str();
    }

   public:
    /// Members a revert needs (Solution_snapshot).
    static constexpr auto snapshot_members() {
        return std::make_tuple(&FUKA_XCTS_Binary_base::space,
                               &FUKA_XCTS_Binary_base::bconfig,
                               &FUKA_XCTS_Binary_base::conformal_factor,
                               &FUKA_XCTS_Binary_base::lapse,
                               &FUKA_XCTS_Binary_base::shift);
    }

    void reset_all_ptrs() override {
        this->reset_support_containers();

        conformal_factor.reset(nullptr);
        lapse.reset(nullptr);
        shift.reset(nullptr);

        space.reset(nullptr);
    }
};

/**
 * @brief Binary solution with neutron-star matter: logarithmic enthalpy and
 * velocity potential after (P, N, beta) in the solution file, the EOS
 * operators' parameter, and the enthalpy rescaling of the hydro-rescaling
 * stages.
 *
 * @tparam space_t binary space (Space_bin_ns, Space_bhns)
 */
template <class space_t>
struct FUKA_XCTS_Binary_matter_base : public FUKA_XCTS_Binary_base<space_t> {
    using binary_base_t = FUKA_XCTS_Binary_base<space_t>;
    using typename binary_base_t::base_config_t;
    using typename binary_base_t::bco_seq_t;
    using typename binary_base_t::bin_res_seq_t;

   protected:
    using binary_base_t::bconfig;
    using binary_base_t::conformal_factor;
    using binary_base_t::lapse;
    using binary_base_t::outputdir;
    using binary_base_t::shift;
    using binary_base_t::space;
    using binary_base_t::syst;

    // EOS parameters
    internal_variable(double, h_cut);
    internal_variable(std::string, eos_file);
    internal_variable(std::string, eos_type);

    ptr_data_member(Scalar, logh, unique);
    ptr_data_member(Scalar, phi, unique);

    /// Parameter of the EOS operators; System_of_eqs::add_ope keeps a
    /// pointer, so it lives as long as the solver.
    Param eos_param;

    /// Unscaled enthalpy of the hydro-rescaling stages (system constant).
    ptr_data_member(Scalar, logh_const, unique);

   protected:
    using binary_base_t::binary_base_t;

    void reset_support_containers() override {
        // refers to the space, rebuilt by the hydro-rescaling setup
        logh_const.reset(nullptr);
        binary_base_t::reset_support_containers();
    }

    void save_solution_to_file() const override {
        bconfig->set_outputdir(outputdir);
        Kadath::bco_utils::save_to_file(*space,
                                        *bconfig,
                                        *conformal_factor,
                                        *lapse,
                                        *shift,
                                        *logh,
                                        *phi);
    }

    void load_matter_fields(FILE* ff) override {
        logh = std::make_unique<Scalar>(*space, ff);
        phi = std::make_unique<Scalar>(*space, ff);
        logh->coef();
        phi->coef();
    }

    /// The hydro-rescaling stages solve for a rescaled enthalpy H: keep it.
    void post_newton_step() override {
        if (logh_const)
            *logh = syst->give_val_def("H");
    }

    /// Converged filename with the EOS name of the first companion.
    std::string matter_converged_filename(std::string const& prefix,
                                          std::string const& stage,
                                          double const M1,
                                          double const M2) const {
        return this->binary_converged_filename(prefix,
                                               stage,
                                               M1,
                                               M2,
                                               extract_eos_name(*bconfig,
                                                                NODES::BCO1));
    }

   public:
    /// Members a revert needs (Solution_snapshot).
    static constexpr auto snapshot_members() {
        return std::tuple_cat(
            binary_base_t::snapshot_members(),
            std::make_tuple(&FUKA_XCTS_Binary_matter_base::logh,
                            &FUKA_XCTS_Binary_matter_base::phi));
    }

    void reset_all_ptrs() override {
        // the system refers to the fields: release it first
        reset_support_containers();
        logh.reset(nullptr);
        phi.reset(nullptr);
        binary_base_t::reset_all_ptrs();
    }
};

/** @}*/
}  // namespace Kadath::FUKA_Solvers
