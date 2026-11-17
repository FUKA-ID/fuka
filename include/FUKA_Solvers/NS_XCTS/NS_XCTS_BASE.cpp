#include "FUKA_Solvers/utilities/solver/solver_utilities.hpp"
#include "bco_utilities.hpp"

namespace Kadath::FUKA_Solvers {

inline NS_XCTS_BASE::NS_XCTS_BASE(NS_XCTS_BASE::base_config_t* config_,
                                  ns_sequence const& seq_,
                                  Parameter_sequence<BCO_PARAMS> const& res_,
                                  std::string outputdir_,
                                  int const rank_)
    : solver_base_t(config_, outputdir_, rank_),
      seq(new ns_sequence(seq_)),
      resolution(new Parameter_sequence<BCO_PARAMS>(res_)) {}

inline void NS_XCTS_BASE::initialize_support_containers() {
    basis.reset(new Base_tensor(shift->get_basis()));
    fmet.reset(new Metric_flat(*space, *basis));

    cfields.reset(new cfgen_t(*space));
    coord_vectors = std::make_unique<cfary_t>(default_co_vector_ary(*space));
}

inline bool NS_XCTS_BASE::is_compatible_solution(base_config_t& old_config) {
    return bconfig->set(BCO_PARAMS::NSHELLS) ==
           old_config.set(BCO_PARAMS::NSHELLS);
}

inline void NS_XCTS_BASE::save_solution_to_file() const {
    bconfig->set_outputdir(outputdir);
    if (diff_omega) {
        bconfig->set_field(Kadath::FUKA_Config::BCO_FIELDS::DIFF_OMEGA) = true;
        Kadath::bco_utils::save_to_file(*space,
                                        *bconfig,
                                        *conformal_factor,
                                        *lapse,
                                        *shift,
                                        *logh,
                                        *diff_omega);
    } else {
        Kadath::bco_utils::save_to_file(*space,
                                        *bconfig,
                                        *conformal_factor,
                                        *lapse,
                                        *shift,
                                        *logh);
    }
}

inline void NS_XCTS_BASE::reset_all_ptrs() {
    reset_support_containers();

    conformal_factor.reset(nullptr);
    lapse.reset(nullptr);
    shift.reset(nullptr);
    logh.reset(nullptr);
    diff_omega.reset(nullptr);

    space.reset(nullptr);
}

inline void NS_XCTS_BASE::regrid() {
    if (static_cast<int>((*bconfig)(BCO_RES)) % 2 == 0)
        throw std::invalid_argument(
            "NS_XCTS::regrid: the resolution must be odd (9, 11, 13, ...)");
    regrid_to("ns_regrid");
}

inline void NS_XCTS_BASE::write_regridded_solution(
    base_config_t& target) const {
    std::cout << "Resolution of old space: "
              << space->get_domain(0)->get_nbr_points()(0) << " (r), "
              << space->get_domain(0)->get_nbr_points()(1) << " (theta), "
              << space->get_domain(0)->get_nbr_points()(2) << " (phi)"
              << std::endl;

    int ndim = 3;
    // get the adapted domain and cast it to its correct type to be able to
    // call its member functions
    const Domain_shell_outer_adapted* old_outer_adapted =
        dynamic_cast<const Domain_shell_outer_adapted*>(space->get_domain(1));

    // setup a scalar field representing the old radius
    Scalar old_space_radius(*space);
    old_space_radius = 0.;

    // get the radius from each domain
    for (int i = 0; i < space->get_nbr_domains(); ++i)
        old_space_radius.set_domain(i) = space->get_domain(i)->get_radius();
    // get the adapted radius of the adapted domain
    old_space_radius.set_domain(1) = old_outer_adapted->get_outer_radius();

    // define a standard decomposition, compatible with the parity of this
    // field
    old_space_radius.std_base();
    // end setup old radius field

    // get the minimal and maximal radius from the adapted domain
    auto [r_min, r_max] = Kadath::bco_utils::get_rmin_rmax(*space, 1);

    std::cout << "Rmin/max: " << r_min << " " << r_max << std::endl;

    // set new resolutions in each spatial dimension
    Dim_array res(ndim);
    res.set(0) = target(BCO_RES);
    res.set(1) = res(0);
    res.set(2) = res(0) - 1;

    std::cout << "Resolution of new space: " << res(0) << " (r), " << res(1)
              << " (theta), " << res(2) << " (phi)" << std::endl;

    // get the type of the colocation points
    int type_coloc = space->get_type_base();

    // Update config
    target.set(BCO_PARAMS::RIN) = 0.5 * r_min;
    target.set(BCO_PARAMS::ROUT) = bco_utils::gold_ratio * r_max;
    target.set(BCO_PARAMS::RMID) = r_max;
    // end update config

    // setup radius bounds of the domains

    int ndom = 4 + target(BCO_PARAMS::NSHELLS);
    std::vector<double> bounds(ndom - 1);
    Kadath::bco_utils::set_NS_bounds(bounds, target);

    Kadath::bco_utils::print_bounds("New bounds: ", bounds);

    // get origin of nucleus domain
    Point center = space->get_domain(0)->get_center();

    // initialize space with new resolution and domain decomposition
    Space_spheric_adapted new_space(type_coloc, center, res, bounds);
    Base_tensor basis(new_space, CARTESIAN_BASIS);

    // get adapted domains to update the radius
    const Domain_shell_outer_adapted* new_outer_adapted =
        dynamic_cast<const Domain_shell_outer_adapted*>(
            new_space.get_domain(1));
    const Domain_shell_inner_adapted* new_inner_adapted =
        dynamic_cast<const Domain_shell_inner_adapted*>(
            new_space.get_domain(2));

    // update adapted domain mapping
    Kadath::bco_utils::interp_adapted_mapping(new_outer_adapted,
                                              1,
                                              old_space_radius);
    Kadath::bco_utils::interp_adapted_mapping(new_inner_adapted,
                                              1,
                                              old_space_radius);

    // setup new fields
    // initialize to one or zero first
    Scalar new_conf(new_space);
    new_conf = 1.;
    new_conf.std_base();

    Scalar new_lapse(new_space);
    new_lapse = 1.;
    new_lapse.std_base();

    Vector new_shift(new_space, CON, basis);
    for (int i = 1; i <= 3; i++)
        new_shift.set(i).annule_hard();
    new_shift.std_base();

    Scalar new_logh(new_space);
    new_logh.annule_hard();
    new_logh.std_base();
    // end setup new fields

    // import data from fields in the old space
    new_conf.import(*conformal_factor);
    new_lapse.import(*lapse);
    new_logh.import(*logh);

    new_shift.set(1).import(shift->set(1));
    new_shift.set(2).import(shift->set(2));
    new_shift.set(3).import(shift->set(3));

    // end import old fields

    // enforce spectral decomposition compatible with the parities
    new_lapse.std_base();
    new_conf.std_base();
    new_logh.std_base();
    new_shift.std_base();

    // Account for Differential rotation
    Scalar new_diff_omega(new_space);
    new_diff_omega.annule_hard();
    if (diff_omega) {
        new_diff_omega.import(*diff_omega);
        new_diff_omega.std_base();
        target.set_field(BCO_FIELDS::DIFF_OMEGA) = true;
        Kadath::bco_utils::save_to_file(new_space,
                                        target,
                                        new_conf,
                                        new_lapse,
                                        new_shift,
                                        new_logh,
                                        new_diff_omega);
    } else {
        target.set_field(BCO_FIELDS::DIFF_OMEGA) = false;
        Kadath::bco_utils::save_to_file(new_space,
                                        target,
                                        new_conf,
                                        new_lapse,
                                        new_shift,
                                        new_logh);
    }
}

inline void NS_XCTS_BASE::load_solution_from_file() {
    std::string spacein{bconfig->space_filename()};
    FILE* ff1 = fopen(spacein.c_str(), "r");

    space.reset(new base_space_t{ff1});
    conformal_factor.reset(new Scalar(*space, ff1));
    lapse.reset(new Scalar(*space, ff1));
    shift.reset(new Vector(*space, ff1));
    logh.reset(new Scalar(*space, ff1));
    lapse->coef();
    shift->coef();
    logh->coef();

    if (bconfig->field(Kadath::FUKA_Config::BCO_FIELDS::DIFF_OMEGA)) {
        diff_omega.reset(new Scalar(*space.get(), ff1));
    } else if (solver_stage == ::Kadath::FUKA_Config::STAGES::DIFF_ROT) {
        diff_omega.reset(new Scalar(*space));
        *diff_omega = ((*bconfig)(BCO_PARAMS::OMEGA) == 0.)
                          ? 1e-7
                          : (*bconfig)(BCO_PARAMS::OMEGA);
        diff_omega->std_base();
    }
    fclose(ff1);

    ndom = space->get_nbr_domains();
}

inline bool NS_XCTS_BASE::increment_resolution() {
    auto const final_res = resolution->final();
    if (!(final_res > resolution->init()) || last_stage_idx != solver_stage)
        return false;

    auto const resolution_indices = resolution->get_indices();
    int const res = (*bconfig)(resolution_indices);
    if (res >= final_res)
        return false;

    bconfig->set(resolution_indices) = next_resolution_step(res, final_res);
    return true;
}

inline bool NS_XCTS_BASE::increment_seq() {
    if (!seq->is_set() || last_stage_idx != solver_stage)
        return false;

    auto sequence_var_indices = seq->get_indices();
    auto const& dx = seq->step_size();
    auto x = bconfig->set(sequence_var_indices) + dx;
    x = (std::fabs(1. - x / seq->final()) < 1e-4) ? seq->final() : x;
    if (seq->loop_condition(x)) {
        bconfig->set(sequence_var_indices) = x;
        return true;
    }
    return false;
}

}  // namespace Kadath::FUKA_Solvers
