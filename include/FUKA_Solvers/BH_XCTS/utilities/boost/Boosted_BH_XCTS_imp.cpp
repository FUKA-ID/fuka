namespace Kadath::FUKA_Solvers {

inline Boosted_BH_XCTS::Boosted_BH_XCTS(base_config_t* config_,
                                        res_seq_t const& res_,
                                        std::string const& outputdir_,
                                        binary_config_t* binary_config_,
                                        int const rank_)
    : BH_XCTS_base(config_, res_, outputdir_, rank_),
      binary_config(binary_config_) {
    solver_stage = STAGES::BIN_BOOST;
    // the converged name identifies the binary the hole was boosted for
    auto& binconfig = *binary_config;
    std::stringstream ss;
    ss << "BIN_BOOST" << "_" << binconfig(DIST) << "_" << binconfig(GOMEGA);
    stagename = ss.str();
    reload();
}

inline void Boosted_BH_XCTS::setup_syst() {
    if (rank == 0)
        std::cout << "############################" << std::endl
                  << "Binary boost using von Neumann BC" << std::endl
                  << "############################" << std::endl;
    bconfig->set(MIRR) =
        Kadath::bco_utils::mirr_from_mch((*bconfig)(CHI), (*bconfig)(MCH));

    update_fields_co(*cfields, *coord_vectors, {}, 0.);
    syst = std::make_unique<System_of_eqs>(*space, 0, ndom - 1);
    syst_init();

    // shift in the frame co-rotating with the binary
    auto& binconfig = *binary_config;
    syst->add_cst("omega_boost", binconfig(GOMEGA));
    syst->add_def("B^i = bet^i + omega_boost * mg^i");

    add_constraint_definitions();
    add_equations("dn(NP) = 0",
                  "bet^i = 0",
                  "B^i = N / P^2 * sm^i + ome * mg^i");
}

}  // namespace Kadath::FUKA_Solvers
