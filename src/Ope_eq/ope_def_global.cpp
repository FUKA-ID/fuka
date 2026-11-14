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


#include "system_of_eqs.hpp"
#include "ope_eq.hpp"
namespace Kadath {
Ope_def_global::Ope_def_global(const System_of_eqs* zesys, int zedom, const char* name_ope) : 
  Ope_eq(zesys, zedom, zesys->get_dom_max() - zesys->get_dom_min() + 1)  {
	for (int n=0 ; n<n_ope ; n++)
	  parts[n] = syst->give_ope (n+syst->get_dom_min(), name_ope) ;
	
	res = new Term_eq (dom, 0., 0.) ;
	auxi = new Term_eq* [n_ope] ;
	for (int i=0 ; i<n_ope ; i++)
	    auxi[i] = 0x0 ;
	compute_res() ;
}

Ope_def_global::~Ope_def_global() {
	delete res ;
	for (int i=0 ; i<n_ope ; i++)
	  if (auxi[i]!=0x0)
	      delete auxi[i] ;
	delete [] auxi ;
}

Term_eq Ope_def_global::compute() const {
	/*if (!called) {
		res = new Term_eq(parts[0]->action()) ;
		called = true ;
	}
	return *res ;*/
	cerr << "Should not call Ope_def_global::action" << endl ;
	abort() ;
}

Term_eq* Ope_def_global::get_res() {
	return res ;
}

void Ope_def_global::compute_res() {
	  
	for (int i=0 ; i<n_ope ; i++)
	    if (auxi[i]==0x0)
	      auxi[i] = new Term_eq (parts[i]->action()) ;
	    else
	      *auxi[i] = parts[i]->action() ;
	    
	bool doder = true ;
	for (int i=0 ; i<n_ope ; i++) 
	  if (auxi[i]->der_d==0x0)
	      doder = false ;
	  
	double val = 0 ;
	for (int i=0 ; i<n_ope ; i++)
	  val += *auxi[i]->val_d ;

	if (!doder) {
	  *res = Term_eq (dom, val) ;
	}
	else {
	  double der = 0 ;
	  for (int i=0 ; i<n_ope ; i++)
	    der += *auxi[i]->der_d ;
	  *res = Term_eq (dom, val, der) ;
	}
}}
