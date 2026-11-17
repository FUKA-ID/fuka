namespace Kadath::FUKA_Solvers {

inline BH_XCTS::BH_XCTS(base_config_t* config_,
                        res_seq_t const& res_,
                        std::string const& outputdir_,
                        int const rank_)
    : BH_XCTS_base(config_, res_, outputdir_, rank_) {
    set_stage(STAGES::TOTAL_BC);
    reload();
}

inline void BH_XCTS::set_stage(STAGES const stage) {
    switch (stage) {
        case STAGES::TOTAL:
            stagename = "TOTAL";
            break;
        case STAGES::TOTAL_BC:
            stagename = "TOTAL_BC";
            break;
        case STAGES::LINBOOST:
            stagename = "LINBOOST";
            break;
        default:
            throw std::invalid_argument("BH_XCTS: unsupported stage");
    }
    solver_stage = stage;
}

inline void BH_XCTS::setup_syst() {
    bconfig->set(MIRR) =
        Kadath::bco_utils::mirr_from_mch((*bconfig)(CHI), (*bconfig)(MCH));

    std::string title{"Total system with von Neumann BC"};
    std::string lapse_bc{"dn(NP) = 0"};
    if (solver_stage == STAGES::TOTAL) {
        title = "Total system with Fixed Lapse BC";
        lapse_bc = "N = n0";
    } else if (solver_stage == STAGES::LINBOOST) {
        title = "Total system with von Neumann BC and linear boost";
    }
    if (rank == 0)
        std::cout << "############################" << std::endl
                  << title << std::endl
                  << "############################" << std::endl;

    update_fields_co(*cfields, *coord_vectors, {}, 0.);
    syst = std::make_unique<System_of_eqs>(*space, 0, ndom - 1);
    syst_init();

    // linear boost at infinity, zero unless LINBOOST
    syst->add_cst("xboost", (*bconfig)(BVELX));
    syst->add_cst("yboost", (*bconfig)(BVELY));
    if (solver_stage == STAGES::TOTAL)
        syst->add_cst("n0", (*bconfig)(FIXED_LAPSE));

    add_constraint_definitions();
    add_equations(lapse_bc,
                  "bet^i = xboost * ex^i + yboost * ey^i",
                  "bet^i = N / P^2 * sm^i + ome * mg^i");
}

}  // namespace Kadath::FUKA_Solvers
