// clang-format off
/*
    Copyright 2017 Philippe Grandclement

    This file is part of Kadath.

    Kadath is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Kadath is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Kadath.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "ope_eq.hpp"
#include "scalar.hpp"
namespace Kadath {
Ope_lap2::Ope_lap2 (const System_of_eqs* zesys, Ope_eq* target) : Ope_eq(zesys, target->get_dom(), 1) {
	parts[0] = target ;
	
}

Ope_lap2::~Ope_lap2() {
}

Term_eq Ope_lap2::compute() const {

	Term_eq target (parts[0]->action()) ;
	int mm ;
        if (target.val_t->is_m_quant_affected())
          mm =  target.val_t->get_parameters()->get_m_quant() ;
        else
          mm = 0 ;
	return target.val_t->get_space().get_domain(dom)->lap2_term_eq(target, mm) ;
}}
