namespace Kadath::FUKA_Solvers {

/**
 * Hydro rescaling: the orbital frequency is fixed and the stellar matter
 * is only rescaled, H = Hconst (1 + Hscale), to keep the baryon mass. Used
 * for the first solution of a new setup (TOTAL_BC with FIXED_GOMEGA) and for
 * the eccentricity reduction (ECC_RED, with the radial velocity adot and a
 * free center of mass).
 */
template <class eos_t>
void BHNS_XCTS<eos_t>::setup_hydro_rescaling_syst() {
    auto& cfg = *bconfig;
    bool const ecc_red = solver_stage == STAGES::ECC_RED;
    if (rank == 0) {
        std::cout << std::string(42, '#') << std::endl
                  << (ecc_red ? "Eccentricity reduction step with fixed "
                                "omega, chi = "
                              : "Hydro Rescaling stage \n"
                                "using fixed orbital velocity with \n"
                                "chi = ")
                  << cfg(BCO_PARAMS::CHI, NODES::BCO1) << ","
                  << cfg(BCO_PARAMS::CHI, NODES::BCO2)
                  << " and q = " << cfg(BIN_PARAMS::Q) << std::endl
                  << std::string(42, '#') << std::endl;
    }
    if (ecc_red)
        binary_syst::set_ecc_red_orbit(cfg,
                                       rank,
                                       cfg(BCO_PARAMS::MADM, NODES::BCO1),
                                       cfg(BCO_PARAMS::MCH, NODES::BCO2));

    // position vector field, only used by ECC_RED
    Vector CART(*space, CON, *basis);
    CART = cfields->cart();

    H_scale = 0;

    // shift of the center of mass along y, may remove x-momenta introduced
    // by the eccentricity reduction
    if (std::isnan(cfg.set(BIN_PARAMS::COMY)))
        cfg.set(BIN_PARAMS::COMY) = 0.;

    // unscaled enthalpy of the previous solution
    logh_const = std::make_unique<Scalar>(*logh);
    logh_const->std_base();

    Kadath::update_fields(*cfields, *coord_vectors, {}, xo, xc1, xc2);

    syst = std::make_unique<System_of_eqs>(*space, 0, ndom - 1);
    syst->add_cst("Hconst", *logh_const);
    syst->add_var("Hscale", H_scale);
    for (int d = 0; d < ndom; ++d) {
        if (d <= space->ADAPTEDNS && d >= space->NS)
            syst->add_def(d, "H  = Hconst * (1. + Hscale)");
        else
            syst->add_def(d, "H  = Hconst");
    }

    Kadath::FUKA_EOS::set_eos_ope_struct<eos_t>()(*syst, eos_param);
    syst_init();

    // the center of mass is fixed for the first solution of a new setup (an
    // integral condition at infinity is unstable there) and solved for by
    // the vanishing ADM momentum in the eccentricity reduction
    if (ecc_red) {
        syst->add_var("xaxis", cfg.set(BIN_PARAMS::COM));
        syst->add_var("yaxis", cfg.set(BIN_PARAMS::COMY));
    } else {
        syst->add_cst("xaxis", cfg(BIN_PARAMS::COM));
        syst->add_cst("yaxis", cfg(BIN_PARAMS::COMY));
    }
    syst->add_cst("ome", cfg(BIN_PARAMS::GOMEGA));
    if (ecc_red)
        binary_syst::add_orbital_shift(*syst,
                                       true,
                                       &cfg.set(BIN_PARAMS::ADOT),
                                       &CART);
    else
        binary_syst::add_orbital_shift(*syst, true);

    bool const corot = cfg.control(CONTROLS::COROT_BIN);
    for (int d = 0; d < ndom; d++) {
        if (d >= space->ADAPTEDNS + 1)
            binary_syst::add_vacuum_eqdefs(*syst, d, corot, false);
        else
            binary_syst::add_matter_eqdefs(*syst, d, corot);
    }
    binary_syst::add_metric_equations(*syst, *space, ndom);

    // the stellar surface is where the matter vanishes
    syst->add_eq_bc(space->ADAPTEDNS, OUTER_BC, "H = 0");
    add_spin_equations();

    space->add_eq_int_volume(*syst,
                             space->NS,
                             space->ADAPTEDNS,
                             "integvolume(intM) = qlMadm1");
    space->add_eq_int_volume(*syst,
                             space->NS,
                             space->ADAPTEDNS,
                             "integvolume(intMb) = Mb1");
    if (ecc_red) {
        space->add_eq_int_inf(*syst, "integ(intPx) = 0");
        space->add_eq_int_inf(*syst, "integ(intPy) = 0");
    }
    add_horizon_equations();
}

}  // namespace Kadath::FUKA_Solvers
