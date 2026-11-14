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
#include "tensor.hpp"
#include "metric.hpp"
#include "system_of_eqs.hpp"
namespace Kadath {
Ope_der_background::Ope_der_background (const System_of_eqs* zesys, int tt, char ind, Ope_eq* target) : Ope_eq(zesys, target->get_dom(), 1), 
		type_der (tt), ind_der(ind) {

	assert ((tt==COV) || (tt==CON)) ;
	parts[0] = target ;
}

Ope_der_background::~Ope_der_background() {
}

Term_eq Ope_der_background::compute() const {
	Term_eq target (parts[0]->action()) ;
	// Check it is a tensor
	if (target.type_data != TERM_T) {
		cerr << "Ope_der_background only defined with respect for a tensor" << endl ;
		abort() ;
	}

	return (syst->get_met()->get_background()->derive(type_der, ind_der, target)) ;
}
}
