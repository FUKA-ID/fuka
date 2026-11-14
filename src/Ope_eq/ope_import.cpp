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
#include "system_of_eqs.hpp"
namespace Kadath {
Ope_import::Ope_import (const System_of_eqs* zesys, int dd, int bb, const char* target) : Ope_eq(zesys, dd), bound(bb),
  others(syst->get_space().get_indices_matching_non_std(dd, bb)) {
  
	n_ope = others.get_size(1) ;
	parts = new Ope_eq* [n_ope] ;
	for (int i=0 ; i<n_ope ; i++)
	  parts[i] = syst->give_ope (others(0, i), target, others(1, i)) ; 
}

Ope_import::~Ope_import() {
}

Term_eq Ope_import::compute() const {

	Term_eq** res = new Term_eq* [n_ope] ;
	for (int i=0 ; i<n_ope ; i++)
	    res[i] = new Term_eq (parts[i]->action()) ;
	
	Term_eq result (syst->get_space().get_domain(dom)->import(dom, bound, n_ope, res)) ;
	for (int i=0 ; i<n_ope ; i++)
	    delete res[i] ;
	delete [] res ;
	
	// Call the member function from domain
	return result ;
}
}
