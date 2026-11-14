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
Ope_partial_var::Ope_partial_var (const System_of_eqs* zesys, int which, Ope_eq* target) : Ope_eq(zesys, target->get_dom(), 1), which_var(which) {
	parts[0] = target ;
}

Ope_partial_var::~Ope_partial_var() {
}

Term_eq Ope_partial_var::compute() const {

	Term_eq target (parts[0]->action()) ;
	// Check it is a tensor
	if (target.type_data != TERM_T) {
		cerr << "Ope_partial_var only defined with respect for a tensor" << endl ;
		abort() ;
	}

	// The value	
	Tensor resval (*target.val_t, false) ;

	for (int i=0 ; i<target.val_t->get_n_comp() ; i++) {
		Array<int> ind (target.val_t->indices(i)) ;
		Val_domain value ((*target.val_t)(ind)(dom)) ;
		if (value.check_if_zero())
			resval.set(ind).set_domain(dom).set_zero() ;
		else {
			Val_domain auxi (value.get_domain()->der_partial_var(value, which_var)) ;
			resval.set(ind).set_domain(dom) = auxi ;
		}
	}

	if (target.der_t!=0x0) {
		Tensor resder (*target.der_t, false) ;
		for (int i=0 ; i<target.der_t->get_n_comp() ; i++) {
			Array<int> ind (target.der_t->indices(i)) ;
			Val_domain value ((*target.der_t)(ind)(dom)) ;
			if (value.check_if_zero())
				resder.set(ind).set_domain(dom).set_zero() ;
			else {
				Val_domain auxi (value.get_domain()->der_partial_var(value, which_var)) ;
				resder.set(ind).set_domain(dom) = auxi ;
			}
			}
		Term_eq res (dom, resval, resder) ;
		return res ;
	}
	else {
		Term_eq res (dom, resval) ;
		return res ;
	}
 }}
