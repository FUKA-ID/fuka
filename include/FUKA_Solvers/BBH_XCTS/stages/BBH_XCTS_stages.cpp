namespace Kadath::FUKA_Solvers {

/**
 * TOTAL_BC: quasi-equilibrium, the orbital frequency fixed by the equality
 * of ADM and Komar masses (integral of dn(N) + 2 dn(P) at infinity), unless
 * FIXED_GOMEGA. ECC_RED: fixed orbital frequency and radial velocity adot.
 * Both: vanishing ADM linear momentum fixes the center-of-mass shifts.
 */
inline void BBH_XCTS::setup_syst() {
    auto& cfg = *bconfig;
    bool const ecc_red = solver_stage == STAGES::ECC_RED;
    bool const fixed_omega = cfg.control(CONTROLS::FIXED_GOMEGA) || ecc_red;
    bool const fixed_lapse = cfg.control(CONTROLS::USE_FIXED_LAPSE);
    bool const corot = cfg.control(CONTROLS::COROT_BIN);
    if (rank == 0) {
        std::cout << std::string(42, '#') << '\n'
                  << (fixed_lapse ? "Total system with fixed Lapse BC"
                                  : "Total system with v.Neumann BC")
                  << '\n';
        if (ecc_red)
            std::cout << "Ecc. Reduction - fixed Omega & Adot" << '\n';
        if (cfg.control(CONTROLS::FIXED_GOMEGA))
            std::cout << "using fixed orbital velocity" << '\n';
        if (corot)
            std::cout << "and co-rotation conditions" << '\n';
    }
    if (ecc_red)
        binary_syst::set_ecc_red_orbit(cfg,
                                       rank,
                                       cfg(BCO_PARAMS::MCH, NODES::BCO1),
                                       cfg(BCO_PARAMS::MCH, NODES::BCO2));

    // position vector field, only used by ECC_RED
    Vector CART(*space, CON, *basis);
    CART = cfields->cart();

    Kadath::update_fields(*cfields, *coord_vectors, {}, xo, xc1, xc2);
    syst = std::make_unique<System_of_eqs>(*space, 0, ndom - 1);
    if (fixed_omega)
        syst->add_cst("ome", cfg(BIN_PARAMS::GOMEGA));
    else
        syst->add_var("ome", cfg.set(BIN_PARAMS::GOMEGA));
    syst->add_var("xaxis", cfg.set(BIN_PARAMS::COM));
    syst->add_var("yaxis", cfg.set(BIN_PARAMS::COMY));

    syst_init();

    std::string bigB{"B^i = bet^i + ome * Morb^i"};
    if (ecc_red) {
        syst->add_cst("adot", cfg(BIN_PARAMS::ADOT));
        syst->add_cst("r", CART);
        syst->add_def("comr^i = r^i - xaxis * ex^i + yaxis * ey^i");
        bigB += " + adot * comr^i";
    }
    syst->add_def(bigB.c_str());

    // vacuum constraint equations
    syst->add_def("eqP     = D^i D_i P + A_ij * A^ij / P^7 / 8");
    syst->add_def("eqNP    = D^i D_i NP - 7. / 8. * NP / P^8 * A_ij * A^ij");
    syst->add_def(
        "eqbet^i = D_j D^j bet^i + D^i D_j bet^j / 3. - 2. * A^ij * D_j "
        "Ntilde");
    binary_syst::add_metric_equations(*syst, *space, ndom);

    if (!fixed_omega)
        space->add_eq_int_inf(*syst, "integ(dn(N) + 2 * dn(P)) = 0");
    space->add_eq_int_inf(*syst, "integ(intPx) = 0");
    space->add_eq_int_inf(*syst, "integ(intPy) = 0");

    if (fixed_lapse) {
        syst->add_cst("nm", cfg(BCO_PARAMS::FIXED_LAPSE, NODES::BCO1));
        syst->add_cst("np", cfg(BCO_PARAMS::FIXED_LAPSE, NODES::BCO2));
    }
    binary_syst::add_apparent_horizon(
        [&](char const* eq) { space->add_bc_sphere_one(*syst, eq); },
        [&](char const* eq) { space->add_eq_int_sphere_one(*syst, eq); },
        "sm",
        fixed_lapse ? "N = nm" : "dn(NP) = 0",
        "integ(intMsq) - Mm * Mm = 0 ");
    binary_syst::add_apparent_horizon(
        [&](char const* eq) { space->add_bc_sphere_two(*syst, eq); },
        [&](char const* eq) { space->add_eq_int_sphere_two(*syst, eq); },
        "sp",
        fixed_lapse ? "N = np" : "dn(NP) = 0",
        "integ(intMsq) - Mp * Mp = 0 ");

    // horizon shift conditions: corotating, or with the holes' spin
    if (corot) {
        space->add_bc_sphere_one(*syst, "B^i = N / P^2 * sm^i");
        space->add_bc_sphere_two(*syst, "B^i = N / P^2 * sp^i");
    } else {
        space->add_bc_sphere_one(*syst, "B^i = N / P^2 * sm^i + ExOme^i");
        if (std::isnan(cfg.set(BCO_PARAMS::FIXED_BCOMEGA, NODES::BCO1)))
            space->add_eq_int_sphere_one(
                *syst,
                "integ(intSm) - chim * CMm * CMm = 0 ");
        space->add_bc_sphere_two(*syst, "B^i = N / P^2 * sp^i + ExOme^i");
        if (std::isnan(cfg.set(BCO_PARAMS::FIXED_BCOMEGA, NODES::BCO2)))
            space->add_eq_int_sphere_two(
                *syst,
                "integ(intSp) - chip * CMp * CMp = 0 ");
    }

    for (int d : excluded_doms()) {
        syst->add_eq_full(d, "N = 0");
        syst->add_eq_full(d, "P = 0");
        syst->add_eq_full(d, "bet^i = 0");
    }
}

}  // namespace Kadath::FUKA_Solvers
