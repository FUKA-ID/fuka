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
Ope_val::Ope_val (const System_of_eqs* zesys, const Index &pp, Ope_eq* target) : 
		Ope_eq(zesys, target->get_dom(), 1),  pos (pp) {
	parts[0] = target ;
}

Ope_val::~Ope_val() {
}

Term_eq Ope_val::compute() const {
	Term_eq target (parts[0]->action()) ;
	// Check it is a tensor
	if (target.type_data != TERM_T) {
		cerr << "Ope_int only defined with respect for a tensor" << endl ;
		abort() ;
	}

	if (target.val_t->get_n_comp() != 1) {
		cerr << "Ope_val only defined with respect to a scalar (yet)" << endl ;
		abort() ;
	}

	// The value
	Array<int> ind (target.val_t->indices(0)) ;
	Val_domain val ((*target.val_t)(ind)(dom)) ;
	val.coef_i() ;
	double resval ;
	if (val.check_if_zero()) 
		resval= 0. ;
	else 
		resval = val(pos) ;

	if (target.der_t!=0x0) {
		Val_domain val_der ((*target.der_t)(ind)(dom)) ;
		val_der.coef_i() ;
		double resder ;
		if (val_der.check_if_zero()) 
			resder = 0. ;
		else 
			resder = val_der(pos) ;
		Term_eq res (dom, resval, resder) ;
		return res ;
	}
	else {
		Term_eq res (dom, resval) ;
		return res ;
	}
}}
