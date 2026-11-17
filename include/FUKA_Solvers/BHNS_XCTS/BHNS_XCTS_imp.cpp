namespace Kadath::FUKA_Solvers {
inline BHNS_XCTS_base::BHNS_XCTS_base(base_config_t* config_,
                                      bco_seq_t const& ns_seq_,
                                      bco_seq_t const& bh_seq_,
                                      bin_res_seq_t const& res_,
                                      std::string outputdir_,
                                      int const rank_)
    : binary_base_t(config_, ns_seq_, bh_seq_, res_, outputdir_, rank_),
      H_scale(0),
      loghc(0) {}

inline std::string BHNS_XCTS_base::converged_filename(
    const std::string stage) const {
    return binary_converged_filename("BHNS",
                                     stage,
                                     (*bconfig)(BCO_PARAMS::MADM, NODES::BCO1),
                                     (*bconfig)(BCO_PARAMS::MCH, NODES::BCO2));
}

inline void BHNS_XCTS_base::update_config_quantities() {
    // the ADM mass of a corotating star at infinite separation is not known:
    // use the quasi-local mass
    if (bconfig->control(CONTROLS::COROT_BIN))
        bconfig->set(BCO_PARAMS::MADM, NODES::BCO1) =
            (*bconfig)(BCO_PARAMS::QLMADM, NODES::BCO1);
}

inline bool BHNS_XCTS_base::increment_resolution() {
    auto const final_res = resolution->final();
    if (!(final_res > resolution->init()))
        return false;

    auto const resolution_indices = resolution->get_indices();
    int const res = (*bconfig)(resolution_indices);
    if (res >= final_res)
        return false;

    bconfig->set(resolution_indices) = next_resolution_step(res, final_res);
    return true;
}

inline void BHNS_XCTS_base::add_spin_equations() {
    if (bconfig->control(CONTROLS::COROT_BIN)) {
        space->add_bc_sphere_two(*syst, "B^i = N / P^2 * sp^i");
        return;
    }
    syst->add_eq_bc(space->ADAPTEDNS, OUTER_BC, "V^i * D_i H = 0");
    binary_syst::add_velocity_potential_eqs(*syst, space->NS, space->ADAPTEDNS);

    if (std::isnan(bconfig->set(BCO_PARAMS::FIXED_BCOMEGA, NODES::BCO1)))
        space->add_eq_int_outer_NS(*syst,
                                   "integ(intS1) / Madm1 / Madm1 = chi1");

    space->add_bc_sphere_two(*syst, "B^i = N / P^2 * sp^i + s^i");
    if (std::isnan(bconfig->set(BCO_PARAMS::FIXED_BCOMEGA, NODES::BCO2)))
        space->add_eq_int_BH(*syst, "integ(intS2) - chi2 * Mch * Mch = 0 ");
}

inline void BHNS_XCTS_base::add_horizon_equations() {
    space->add_bc_sphere_two(*syst, "sp^j * D_j NP = 0");
    space->add_bc_sphere_two(
        *syst,
        "sp^j * D_j P + P / 4 * D^j sp_j + A_ij * sp^i * sp^j / P^3 / 4 = 0");
    space->add_eq_int_BH(*syst, "integ(intMsq) - Mirr * Mirr = 0 ");

    for (int d : excluded_doms()) {
        syst->add_eq_full(d, "N = 0");
        syst->add_eq_full(d, "P = 0");
        syst->add_eq_full(d, "bet^i = 0");
    }
}

}  // namespace Kadath::FUKA_Solvers
