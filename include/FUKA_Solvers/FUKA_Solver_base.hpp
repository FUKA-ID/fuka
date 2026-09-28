/*
 * Copyright 2022
 * This file is part of the KADATH library and published under
 * https://arxiv.org/abs/2103.09911
 *
 * Author:
 * Samuel D. Tootle <tootle@itp.uni-frankfurt.de>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */
#pragma once
#include <cstdlib>
#include <string>

#include "Configurator/config_bco.hpp"
#include "Configurator/config_binary.hpp"
#include "EOS/EOS.hh"
#include "bco_utilities.hpp"
#include "coord_fields.hpp"

#include "codes_utilities.hpp"
#include "kadath.hpp"
#include "kadath_adapted.hpp"
#include "name_tools.hpp"

#include "FUKA_Solvers/utilities/config_utils.hpp"
#include "FUKA_Solvers/utilities/solver_utilities.hpp"

#include "Solvers/sequences/parameter_sequence.hpp"

#if defined __cpp_lib_filesystem && __cpp_lib_filesystem < 201703L
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#else
#include <filesystem>
namespace fs = std::filesystem;
#endif

/** \addtogroup Solver_base
 * \ingroup FUKA
 * Various FUKA initial data solvers
 * @{
 */

namespace Kadath::FUKA_Solvers {

using namespace ::Kadath::FUKA_Config;
using namespace ::Kadath::FUKA_Config_Utils;

struct FUKA_Solver_base {
   protected:
    internal_variable(int, rank)
    internal_variable(int, verbosity)
    internal_variable(int, ndom)
    internal_variable(int, last_stage_idx)
    internal_variable(std::string, outputdir)
    internal_variable(std::string, stagename)
    internal_variable(STAGES, solver_stage);

    virtual ~FUKA_Solver_base() = default;

    FUKA_Solver_base()
        : rank(0),
          verbosity(0),
          ndom(-1),
          last_stage_idx(STAGES::NUM_STAGES),
          solver_stage(STAGES::NUM_STAGES),
          outputdir("./"),
          stagename("") {};

    FUKA_Solver_base(int const rank_,
                     int const verbosity_,
                     int const ndom_,
                     int const last_stage_idx_,
                     std::string const& outputdir_,
                     std::string const& stagename_)
        : rank(rank_),
          verbosity(verbosity_),
          ndom(ndom_),
          last_stage_idx(last_stage_idx_),
          solver_stage(STAGES::NUM_STAGES),
          outputdir(outputdir_),
          stagename(stagename_) {};

   protected:
    // I/O Methods
    virtual std::string converged_filename(const std::string stage) const = 0;
    virtual void save_solution_to_file() const = 0;
    virtual void load_solution_from_file() = 0;
    virtual void initialize_support_containers() = 0;
    virtual void update_config_quantities() = 0;
    virtual void print_diagnostics(const int ite, const double conv) = 0;

   public:
    virtual void reset_all_ptrs() = 0;

    // Regrid methodss
    virtual bool increment_resolution() = 0;
    virtual void regrid() = 0;

    // Solver methods
    virtual void syst_init() = 0;
    virtual int do_newton() = 0;

   protected:
    void reload() {
        reset_all_ptrs();
        load_solution_from_file();
        initialize_support_containers();
    }
};

/** @}*/
}  // namespace Kadath::FUKA_Solvers