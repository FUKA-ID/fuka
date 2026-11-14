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
Ope_inverse::Ope_inverse (const System_of_eqs* zesys, Ope_eq* target) : Ope_eq(zesys, target->get_dom(), 1) {
	parts[0] = target ;
}

Ope_inverse::~Ope_inverse() {
}

Term_eq Ope_inverse::compute() const {

	Term_eq target (parts[0]->action()) ;
	
	// Various checks
	if (target.type_data != TERM_T) {
		cerr << "Ope_inverse only defined with respect for a tensor" << endl ;
		abort() ;
	}
	if (target.val_t->get_valence()!=2) {
		cerr << "Ope_inverse only defined with respect to a second order tensor" << endl ;
		abort() ;
	}
	if (target.val_t->get_index_type(0)!=target.val_t->get_index_type(1)) {
		cerr << "Ope_inverse only defined with respect to a tensor which indices are of the same type" << endl ;
		abort() ;
	}
	int typeresult = - target.val_t->get_index_type(0) ;
	int dim = target.val_t->get_space().get_ndim() ;

	bool m_doder = (target.der_t==0x0) ? false : true ;

	 Term_eq** res(new Term_eq* [dim*(dim + 1)/2]);                // 6 components to compute...
   Scalar val(target.val_t->get_space());                       // val and cmpval are more or less the same intermediate, but with different types
   Scalar der(target.val_t->get_space());
   Val_domain cmpval(target.val_t->get_space().get_domain(dom));
   Val_domain cmpder(target.val_t->get_space().get_domain(dom));

   Val_domain detval(target.val_t->get_space().get_domain(dom));     // let's compute the determinant...
   Val_domain detder(target.val_t->get_space().get_domain(dom));     // ... and its automatic derivative
   detval = 0.0;
   detder = 0.0;

   // compute determinant as a sum over permutations
   double signature;                                 // signature of the permutation
   gsl_permutation* p(gsl_permutation_calloc(dim));  // initialise to identity 0, 1, 2
   do
   {
      signature = sign(p);
      int p1(static_cast<int>(gsl_permutation_get(p, 0)) + 1);   // for tensor indices, you need to add one to get a permutation of 1, 2, 3
      int p2(static_cast<int>(gsl_permutation_get(p, 1)) + 1);
      int p3(static_cast<int>(gsl_permutation_get(p, 2)) + 1);
      detval += signature * (*target.val_t)(p1, 1)(dom) * (*target.val_t)(p2, 2)(dom) * (*target.val_t)(p3, 3)(dom);   // sum on permutations
      if (m_doder)
      {
         detder += signature *((*target.der_t)(p1, 1)(dom) * (*target.val_t)(p2, 2)(dom) * (*target.val_t)(p3, 3)(dom)
                             + (*target.val_t)(p1, 1)(dom) * (*target.der_t)(p2, 2)(dom) * (*target.val_t)(p3, 3)(dom)
                             + (*target.val_t)(p1, 1)(dom) * (*target.val_t)(p2, 2)(dom) * (*target.der_t)(p3, 3)(dom));
      }
   }while(gsl_permutation_next(p) == GSL_SUCCESS);
   gsl_permutation_free(p);

   // compute the transposed comatrix
   p = gsl_permutation_calloc(dim - 1);  // initialise to identity 0, 1
   for (int i(1) ; i <= dim ;  ++i)      // sum over components, only the upper triangular part
   {
      for (int j(i) ; j <= dim ; ++j)
      {
         cmpval = 0.0;                   // initialise to 0 before each computation
         cmpder = 0.0;                   // initialise to 0 before each computation
         do
         {
            signature = sign(p);
            int p1(static_cast<int>(gsl_permutation_get(p, 0)) + 1);  // always add one for tensor indices, now we have a permutation of 1, 2
            int p2(static_cast<int>(gsl_permutation_get(p, 1)) + 1);
            vector<int> index1, index2;                      // manage the indices of the transposed comatrix
            index1 = ind_com(i, j, p1, 1);                   // we need to play a bit with indices to get transpose(COM(metric))
            index2 = ind_com(i, j, p2, 2);
            cmpval += pow(-1.0, i + j)*signature*(*target.val_t)(index1[0], index1[1])(dom)*(*target.val_t)(index2[0],index2[1])(dom);      // sum on permutations
            if (m_doder)
               cmpder += pow(-1.0, i + j)*signature*(*target.der_t)(index1[0], index1[1])(dom)*(*target.val_t)(index2[0], index2[1])(dom)
                      +  pow(-1.0, i + j)*signature*(*target.val_t)(index1[0], index1[1])(dom)*(*target.der_t)(index2[0], index2[1])(dom);
         }while(gsl_permutation_next(p) == GSL_SUCCESS);
	 gsl_permutation_free(p) ;
         p = gsl_permutation_calloc(dim - 1);               // reinitialise to identity 0, 1, ready to be used for next component
         val.set_domain(dom) = cmpval / detval;              // now we have the component of the inverse matrix
         if (m_doder)
         {
            der.set_domain(dom) = cmpder / detval - cmpval * detder / (detval*detval);
            res[j - 1 + dim*(i - 1) - i*(i - 1)/2] = new Term_eq (dom, val, der);   // just save upper triangular components in a 1D array of Term_eq (the indice will run throug 0, 1, 2, 3, 4, 5 with i, j)
         }
         else
            res[j - 1 + dim*(i - 1) - i*(i - 1)/2] = new Term_eq (dom, val);
      }
   }
   gsl_permutation_free(p);

   // Value field :
   Metric_tensor resval(target.val_t->get_space() , typeresult, target.val_t->get_basis());
   Metric_tensor resder(target.val_t->get_space() , typeresult, target.val_t->get_basis());
   for (int i(1) ; i <= dim ; ++i)
      for (int j(i) ; j <= dim ; ++j)
         resval.set(i, j) = res[j - 1 + dim*(i - 1) - i*(i - 1)/2]->get_val_t();

   // Der field
   if (m_doder) {
      for (int i(1) ; i <= dim ; ++i)
         for (int j(i) ; j <= dim ; ++j)
            resder.set(i, j) = res[j - 1 + dim*(i - 1) - i*(i - 1)/2]->get_der_t();
   }

   for (int i(0) ; i < dim*(dim + 1)/2 ; ++i)
      delete res[i];
   delete[] res;	

   if (!m_doder) 
	return Term_eq (dom, resval) ;
   else
	return Term_eq (dom, resval, resder) ;
}
}
