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
#include "system_of_eqs.hpp"
namespace Kadath {
Ope_point::Ope_point (const System_of_eqs* zesys, const Point &MM, Ope_eq* target) : 
		Ope_eq(zesys, target->get_dom(), 1),  num(MM){

	parts[0] = target ;
	num = zesys->get_space().get_domain(dom)->absol_to_num(MM) ;	
}

Ope_point::~Ope_point() {
}

Term_eq Ope_point::compute() const {
	Term_eq target (parts[0]->action()) ;
	
	// Check it is a tensor
	if (target.type_data != TERM_T) {
		cerr << "Ope_point only defined with respect for a tensor" << endl ;
		abort() ;
	}

	if (target.val_t->get_n_comp() != 1) {
		cerr << "Ope_point only defined with respect to a scalar (yet)" << endl ;
		abort() ;
	}

	
	// The value
	Array<int> ind (target.val_t->indices(0)) ;
	Val_domain val ((*target.val_t)(ind)(dom)) ;
	
	double resval ;
	if (val.check_if_zero()) 
		resval= 0. ;
	else  {
	    val.coef() ;
	    resval = val.get_base().summation(num, val.get_coef()) ;
	}
	if (target.der_t!=0x0) {
		Array<int> indder (target.der_t->indices(0)) ;
		Val_domain valder ((*target.der_t)(ind)(dom)) ;
		
		double resder ;
		if (valder.check_if_zero()) 
		    resder= 0. ;
		else {
		    valder.coef() ;
		    resder = valder.get_base().summation(num, valder.get_coef()) ;
		}
		Term_eq res (dom, resval, resder) ;
		return res ;
	}
	else {
		Term_eq res (dom, resval) ;
		return res ;
	}
}}
