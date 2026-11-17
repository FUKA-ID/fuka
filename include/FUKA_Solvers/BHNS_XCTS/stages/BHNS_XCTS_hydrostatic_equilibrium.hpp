namespace Kadath::FUKA_Solvers {

/**
 * Hydrostatic equilibrium (TOTAL_BC): the star satisfies the first integral
 * of the Euler equation; the orbital frequency and the center of mass follow
 * from the force balance at the stellar center and the vanishing ADM
 * momentum.
 */
template <class eos_t>
void BHNS_XCTS<eos_t>::setup_hydrostatic_equilibrium_syst() {
    auto& cfg = *bconfig;
    if (rank == 0)
        std::cout << std::string(42, '#') << std::endl
                  << "TOTAL_BC - Hydrostatic equilibrium stage\n"
                  << "using v.Neumann lapse BC\n"
                  << std::string(42, '#') << std::endl;

    loghc = Kadath::bco_utils::get_boundary_val(space->NS, *logh, INNER_BC);
    Kadath::update_fields(*cfields, *coord_vectors, {}, xo, xc1, xc2);
    logh_const.reset(nullptr);

    syst = std::make_unique<System_of_eqs>(*space, 0, ndom - 1);
    syst->add_var("H", *logh);

    Kadath::FUKA_EOS::set_eos_ope_struct<eos_t>()(*syst, eos_param);
    syst_init();

    syst->add_var("ome", cfg.set(BIN_PARAMS::GOMEGA));
    syst->add_var("xaxis", cfg.set(BIN_PARAMS::COM));
    syst->add_var("yaxis", cfg.set(BIN_PARAMS::COMY));

    // central log enthalpy, fixed by the baryon mass
    syst->add_var("Hc1", loghc);

    binary_syst::add_orbital_shift(*syst, true);

    bool const corot = cfg.control(CONTROLS::COROT_BIN);
    for (int d = 0; d < ndom; d++) {
        if (d >= space->ADAPTEDNS + 1)
            binary_syst::add_vacuum_eqdefs(*syst, d, corot, true);
        else
            binary_syst::add_matter_eqdefs(*syst, d, corot);
    }
    binary_syst::add_metric_equations(*syst, *space, ndom);

    // the stellar surface is where the matter vanishes
    syst->add_eq_bc(space->ADAPTEDNS, OUTER_BC, "H = 0");
    add_spin_equations();

    // force balance at the stellar center fixes the orbital frequency
    Index posori1(space->get_domain(space->NS)->get_nbr_points());
    syst->add_eq_val(space->NS, "ex^i * D_i H", posori1);

    syst->add_eq_first_integral(space->NS,
                                space->ADAPTEDNS,
                                "firstint",
                                "H - Hc1");

    space->add_eq_int_volume(*syst,
                             space->NS,
                             space->ADAPTEDNS,
                             "integvolume(intM) = qlMadm1");
    space->add_eq_int_volume(*syst,
                             space->NS,
                             space->ADAPTEDNS,
                             "integvolume(intMb) = Mb1");

    // vanishing ADM momentum fixes the center of mass
    space->add_eq_int_inf(*syst, "integ(intPy) = 0");
    space->add_eq_int_inf(*syst, "integ(intPx) = 0");

    add_horizon_equations();
}

}  // namespace Kadath::FUKA_Solvers
