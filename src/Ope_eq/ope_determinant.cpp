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
Ope_determinant::Ope_determinant (const System_of_eqs* zesys, Ope_eq* target) : Ope_eq(zesys, target->get_dom(), 1) {
	parts[0] = target ;
}

Ope_determinant::~Ope_determinant() {
}

Term_eq Ope_determinant::compute() const {

	Term_eq target (parts[0]->action()) ;
	
	// Various checks
	if (target.type_data != TERM_T) {
		cerr << "Ope_determinant only defined with respect for a tensor" << endl ;
		abort() ;
	}
	if (target.val_t->get_valence()!=2) {
		cerr << "Ope_determinant only defined with respect to a second order tensor" << endl ;
		abort() ;
	}
	if (target.val_t->get_index_type(0)!=target.val_t->get_index_type(1)) {
		cerr << "Ope_determinant only defined with respect to a tensor which indices are of the same type" << endl ;
		abort() ;
	}
	
	int ndim = target.val_t->get_space().get_ndim() ;

	switch (ndim) {
		case 2 :{
			  
		  Val_domain resval ((*target.val_t)(1,1)(dom)* (*target.val_t)(2,2)(dom) - 
			  (*target.val_t)(1,2)(dom)* (*target.val_t)(2,1)(dom)) ;
		  Scalar res (target.val_t->get_space()) ;
		  res.set_domain(dom) = resval ;
			  
		  if (target.der_t != 0x0) {
		    Val_domain derval ((*target.val_t)(1,1)(dom)* (*target.der_t)(2,2)(dom) + 
		      (*target.der_t)(1,1)(dom)* (*target.val_t)(2,2)(dom) -  
		      (*target.val_t)(1,2)(dom)* (*target.der_t)(2,1)(dom) - 
		      (*target.der_t)(1,2)(dom)* (*target.val_t)(2,1)(dom)) ;
		      
		     Scalar der (target.val_t->get_space()) ;
		     der.set_domain(dom) = derval ;
		     
		     return Term_eq (dom, res, der) ;
		  }
		  else {
		      return Term_eq (dom, res) ;	  
		  }		
		}
		case 3 : {
		  
		  
			Val_domain resval ((*target.val_t)(1,1)(dom)* (*target.val_t)(2,2)(dom)* (*target.val_t)(3,3)(dom)
			+ (*target.val_t)(2,1)(dom)* (*target.val_t)(3,2)(dom)* (*target.val_t)(1,3)(dom)
			+ (*target.val_t)(3,1)(dom)* (*target.val_t)(1,2)(dom)* (*target.val_t)(2,3)(dom) 
			- (*target.val_t)(1,3)(dom)* (*target.val_t)(2,2)(dom)* (*target.val_t)(3,1)(dom)
			- (*target.val_t)(2,3)(dom)* (*target.val_t)(3,2)(dom)* (*target.val_t)(1,1)(dom)
			- (*target.val_t)(3,3)(dom)* (*target.val_t)(1,2)(dom)* (*target.val_t)(2,1)(dom)) ;
			Scalar res (target.val_t->get_space()) ;
			res.set_domain(dom) = resval ;
			
			//if (target.val_t->get_basis().get_basis(dom)==SPHERICAL_BASIS) {
			//	res.affect_parameters() ;
			//	res.set_parameters()->set_m_order() = 2 ;
			//}
			
			if (target.der_t!=0x0) {
			    Val_domain derval ((*target.der_t)(1,1)(dom)* (*target.val_t)(2,2)(dom)* (*target.val_t)(3,3)(dom) 
			    + (*target.val_t)(1,1)(dom)* (*target.der_t)(2,2)(dom)* (*target.val_t)(3,3)(dom)
			    + (*target.val_t)(1,1)(dom)* (*target.val_t)(2,2)(dom)* (*target.der_t)(3,3)(dom)
			    + (*target.der_t)(2,1)(dom)* (*target.val_t)(3,2)(dom)* (*target.val_t)(1,3)(dom) 
			    + (*target.val_t)(2,1)(dom)* (*target.der_t)(3,2)(dom)* (*target.val_t)(1,3)(dom)
			    + (*target.val_t)(2,1)(dom)* (*target.val_t)(3,2)(dom)* (*target.der_t)(1,3)(dom)
			    + (*target.der_t)(3,1)(dom)* (*target.val_t)(1,2)(dom)* (*target.val_t)(2,3)(dom) 
			    + (*target.val_t)(3,1)(dom)* (*target.der_t)(1,2)(dom)* (*target.val_t)(2,3)(dom)
			    + (*target.val_t)(3,1)(dom)* (*target.val_t)(1,2)(dom)* (*target.der_t)(2,3)(dom)
			    - (*target.der_t)(1,3)(dom)* (*target.val_t)(2,2)(dom)* (*target.val_t)(3,1)(dom) 
			    - (*target.val_t)(1,3)(dom)* (*target.der_t)(2,2)(dom)* (*target.val_t)(3,1)(dom)
			    - (*target.val_t)(1,3)(dom)* (*target.val_t)(2,2)(dom)* (*target.der_t)(3,1)(dom)
			    - (*target.der_t)(2,3)(dom)* (*target.val_t)(3,2)(dom)* (*target.val_t)(1,1)(dom) 
			    - (*target.val_t)(2,3)(dom)* (*target.der_t)(3,2)(dom)* (*target.val_t)(1,1)(dom)
			    - (*target.val_t)(2,3)(dom)* (*target.val_t)(3,2)(dom)* (*target.der_t)(1,1)(dom)
			    - (*target.der_t)(3,3)(dom)* (*target.val_t)(1,2)(dom)* (*target.val_t)(2,1)(dom) 
			    - (*target.val_t)(3,3)(dom)* (*target.der_t)(1,2)(dom)* (*target.val_t)(2,1)(dom)
			    - (*target.val_t)(3,3)(dom)* (*target.val_t)(1,2)(dom)* (*target.der_t)(2,1)(dom)) ;
			    
			    Scalar der (target.val_t->get_space()) ;
			    der.set_domain(dom) = derval ;
			   
			   // if (target.val_t->get_basis().get_basis(dom)==SPHERICAL_BASIS) {
			//		der.affect_parameters() ;
				//	der.set_parameters()->set_m_order() = 2 ;
				///}
			    return Term_eq (dom, res, der) ;
			}
			else
			  return Term_eq (dom, res) ;
			}
		default:
			cerr << "Ope_determinant not implemented in " << ndim << " dimensions" << endl ;
			abort() ;
	}
}}
