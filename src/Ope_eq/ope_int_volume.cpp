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
Ope_int_volume::Ope_int_volume (const System_of_eqs* zesys, Ope_eq* target) : Ope_eq(zesys, target->get_dom(), 1) {
	parts[0] = target ;
}

Ope_int_volume::~Ope_int_volume() {
}

Term_eq Ope_int_volume::compute() const {

	Term_eq target (parts[0]->action()) ;
	if (target.get_type_data()==TERM_T) 
	  return target.val_t->get_space().get_domain(dom)->integ_volume_term_eq(target) ;
	else {
	    assert (target.get_type_data()==TERM_D) ;
	    Scalar auxival (syst->get_space()) ;
	    auxival.set_domain(dom) = target.get_val_d() ;
	    auxival.set_domain(dom).std_base() ;
	    if (target.der_d==0x0) {
		Term_eq auxi (dom, auxival) ;
		return auxi.val_t->get_space().get_domain(dom)->integ_volume_term_eq(auxi) ;
	    }
	    else {
	        Scalar auxider (syst->get_space()) ;
		auxider.set_domain(dom) = target.get_der_d() ;
		auxider.set_domain(dom).std_base() ;
		Term_eq auxi (dom, auxival, auxider) ;
		return auxi.val_t->get_space().get_domain(dom)->integ_volume_term_eq(auxi) ;
	    }
	}
}
}
