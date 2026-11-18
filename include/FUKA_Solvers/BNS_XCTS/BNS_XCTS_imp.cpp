namespace Kadath::FUKA_Solvers {
inline BNS_XCTS_base::BNS_XCTS_base(base_config_t* config_,
                                    bco_seq_t const& ns1_seq_,
                                    bco_seq_t const& ns2_seq_,
                                    bin_res_seq_t const& res_,
                                    std::string outputdir_,
                                    int const rank_)
    : binary_base_t(config_, ns1_seq_, ns2_seq_, res_, outputdir_, rank_),
      H_scale1(0),
      H_scale2(0) {}

inline std::unique_ptr<BNS_XCTS_base::base_space_t> BNS_XCTS_base::read_space(
    FILE* ff) const {
    return std::make_unique<base_space_t>(ff,
                                          !bconfig->control(CONTROLS::NEW_ID));
}

inline std::string BNS_XCTS_base::converged_filename(
    const std::string stage) const {
    return matter_converged_filename("BNS",
                                     stage,
                                     bco2_seq->get_mass_fixing_val(),
                                     bco1_seq->get_mass_fixing_val());
}

}  // namespace Kadath::FUKA_Solvers
