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

#include "space.hpp"
#include "system_of_eqs.hpp"
#include "name_tools.hpp"
#include "scalar.hpp"
namespace Kadath {
// Constructor
System_of_eqs::System_of_eqs (const Space& sp) : espace(sp), dom_min(0), 
		dom_max(espace.get_nbr_domains()-1), ndom(dom_max-dom_min+1),
		nvar_double (0), nvar(0), nterm_double(0), assoc_var_double(VARMAX*ndom), nterm(0), assoc_var(VARMAX*ndom), 
		ncst(0), nterm_cst(0), ncst_hard(0), val_cst_hard(VARMAX), ndef(0), ndef_glob(0),  nopeuser(0), nopeuser_bin(0),
		met(0x0), name_met(0x0),
		neq_int(0),
		neq(0), nbr_unknowns(0), nbr_conditions(-1), which_coef(0x0) {

	nbr_unknowns = sp.nbr_unknowns_from_variable_domains() ;
		  
	var_double = new double* [VARMAX] ;
	names_var_double = new char*[VARMAX] ;

	term_double = new Term_eq* [VARMAX*ndom] ;

	var = new Tensor* [VARMAX] ;
	names_var = new char*[VARMAX] ;
	
	term = new Term_eq* [VARMAX*ndom] ;

	cst = new Term_eq* [VARMAX*ndom] ;
	names_cst = new char*[VARMAX*ndom] ;

	cst_hard = new Term_eq* [VARMAX] ;

	def = new Ope_def* [VARMAX*ndom] ;
	names_def = new char*[VARMAX*ndom] ;

	def_glob = new Ope_def_global* [VARMAX*ndom] ;
	names_def_glob = new char*[VARMAX*ndom] ;

	names_opeuser = new char*[VARMAX] ;
	paruser = new Param* [VARMAX] ;

	names_opeuser_bin = new char*[VARMAX] ;
	paruser_bin = new Param* [VARMAX] ;

	eq_int = new Eq_int* [VARMAX] ;

	eq = new Equation* [VARMAX] ;
	results = new Term_eq* [VARMAX] ;

	for (int i=0 ; i<VARMAX ; i++) {
		var_double[i] = 0x0 ;
		var[i]=0x0 ;
		names_var[i]=0x0 ;
		
		cst_hard[i] = 0x0 ;
		names_cst[i] = 0x0 ;

		opeuser[i] = 0x0 ;
		paruser[i] = 0x0 ;
		
		opeuser_bin[i] = 0x0 ;
		paruser_bin[i] = 0x0 ;
		
		eq_int[i] = 0x0 ;
		
		eq[i] = 0x0 ;
		results[i] = 0x0 ;
		}		
	for (int i=0 ; i<VARMAX*ndom ; i++) {
		term[i] = 0x0 ;
		cst[i] = 0x0 ;
		term_double[i] = 0x0 ;
	}
}

// same between two bounds
System_of_eqs::System_of_eqs (const Space& sp, int dmin, int dmax) : espace(sp), dom_min(dmin), 
		dom_max(dmax), ndom(dom_max-dom_min+1),
		nvar_double(0), nvar(0), nterm_double(0), assoc_var_double(VARMAX*ndom),
		nterm(0), assoc_var(VARMAX*ndom), 
		ncst(0),nterm_cst(0), ncst_hard(0), 
		val_cst_hard(VARMAX), ndef(0), ndef_glob(0), nopeuser(0), nopeuser_bin(0), met(0x0), name_met(0x0),
		neq_int(0), neq(0), nbr_unknowns(0), nbr_conditions(-1), which_coef(0x0) {
	nbr_unknowns = sp.nbr_unknowns_from_variable_domains() ;
	var_double = new double* [VARMAX] ;
	names_var_double = new char*[VARMAX] ;

	term_double = new Term_eq* [VARMAX*ndom] ;
	
	var = new Tensor* [VARMAX] ;
	names_var = new char*[VARMAX] ;

	term = new Term_eq* [VARMAX*ndom] ;

	cst = new Term_eq* [VARMAX*ndom] ;
	names_cst = new char*[VARMAX*ndom] ;

	cst_hard = new Term_eq* [VARMAX] ;

	def = new Ope_def* [VARMAX*ndom] ;
	names_def = new char*[VARMAX*ndom] ;

	def_glob = new Ope_def_global* [VARMAX*ndom] ;
	names_def_glob = new char*[VARMAX*ndom] ;

	names_opeuser = new char*[VARMAX] ;
	paruser = new Param* [VARMAX] ;

	names_opeuser_bin = new char*[VARMAX] ;
	paruser_bin = new Param* [VARMAX] ;

	eq_int = new Eq_int* [VARMAX] ;

	eq = new Equation* [VARMAX] ;
	results = new Term_eq* [VARMAX] ;

	for (int i=0 ; i<VARMAX ; i++) {
		var_double[i] = 0x0 ;
		var[i]=0x0 ;
		names_var[i]=0x0 ;
		cst_hard[i] = 0x0 ;
		names_cst[i] = 0x0 ;
		opeuser[i] = 0x0 ;
		paruser[i] = 0x0 ;
		opeuser_bin[i] = 0x0 ;
		paruser_bin[i] = 0x0 ;
		eq_int[i] = 0x0 ;
		eq[i] = 0x0 ;
		results[i] = 0x0 ;
		}		
	for (int i=0 ; i<VARMAX*ndom ; i++) {
		term[i] = 0x0 ;
		cst[i] = 0x0 ;
		term_double[i] = 0x0 ;
	}	
}

// In only one domain
System_of_eqs::System_of_eqs (const Space& sp, int dd) : espace(sp), dom_min(dd), 
		dom_max(dd), ndom(1),
		nvar_double(0), nvar(0), nterm_double(0), assoc_var_double(VARMAX*ndom), 
		nterm(0), assoc_var(VARMAX), 
		ncst(0), nterm_cst(0), ncst_hard(0), val_cst_hard(VARMAX), ndef(0), ndef_glob(0), nopeuser(0),nopeuser_bin(0),
		met(0x0), name_met(0x0),
		neq_int(0), neq(0), nbr_unknowns(0), nbr_conditions(-1), which_coef(0x0) {
nbr_unknowns = sp.nbr_unknowns_from_variable_domains() ;
	var_double = new double* [VARMAX] ;
	names_var_double = new char*[VARMAX] ;

	term_double = new Term_eq* [VARMAX*ndom] ;

	var = new Tensor* [VARMAX] ;
	names_var = new char*[VARMAX] ;

	term = new Term_eq* [VARMAX*ndom] ;

	cst = new Term_eq* [VARMAX*ndom] ;
	names_cst = new char*[VARMAX*ndom] ;

	cst_hard = new Term_eq* [VARMAX] ;

	def = new Ope_def* [VARMAX*ndom] ;
	names_def = new char*[VARMAX*ndom] ;
	
	def_glob = new Ope_def_global* [VARMAX*ndom] ;
	names_def_glob = new char*[VARMAX*ndom] ;

	names_opeuser = new char* [VARMAX] ;
	paruser = new Param* [VARMAX]  ;

	names_opeuser_bin = new char* [VARMAX] ;
	paruser_bin = new Param* [VARMAX]  ;

	eq_int = new Eq_int* [VARMAX] ;

	eq = new Equation* [VARMAX] ;
	results = new Term_eq* [VARMAX] ;

	for (int i=0 ; i<VARMAX ; i++) {
		var_double[i]=0x0 ;
		var[i]=0x0 ;
		names_var[i]=0x0 ;
		cst_hard[i] = 0x0 ;
		names_cst[i] = 0x0 ;
		opeuser[i] =  0x0 ;
		paruser[i] = 0x0 ;
		opeuser_bin[i] =  0x0 ;
		paruser_bin[i] = 0x0 ;
		eq_int[i] = 0x0 ;
		eq[i] = 0x0 ;
		results[i] = 0x0 ;
		}		
	for (int i=0 ; i<VARMAX*ndom ; i++) {
		term[i] = 0x0 ;
		cst[i] = 0x0 ;
		term_double[i] = 0x0 ;
	}	
}

System_of_eqs::System_of_eqs (const System_of_eqs& so) : espace(so.espace), assoc_var_double(0), assoc_var(0), val_cst_hard(0) {
	cerr << "Copy constructor not implemented for System_of_eqs" << endl ;
	abort() ;
}

System_of_eqs::~System_of_eqs() {	

	delete [] var_double ;
	for (int i=0 ; i<nvar_double ; i++)
		delete [] names_var_double[i] ;
	delete [] names_var_double ;

	for (int i=0 ; i<nterm_double ; i++)
		delete term_double[i] ;
	delete [] term_double ;

	delete [] var ;
	for (int i=0 ; i<nvar ; i++)
		delete [] names_var[i] ;
	delete [] names_var ;

	for (int i=0 ; i<nterm ; i++)
		delete term[i] ;
	delete [] term ;

	for (int i=0 ; i<nterm_cst ; i++) {
		delete cst[i] ;
		cst[i] = 0x0 ;	
	}
	delete [] cst ;

	for (int i=0 ; i<ncst ; i++)
		delete [] names_cst[i] ;
	delete [] names_cst ;

	for (int i=0 ; i<ncst_hard ; i++)
		delete cst_hard[i] ;
	delete [] cst_hard ;
	
	for (int i=0 ; i<ndef ; i++)
		delete def[i] ;
	delete [] def ;

	for (int i=0 ; i<ndef ; i++)
		delete [] names_def[i] ;
	delete [] names_def ;
	
      for (int i=0 ; i<ndef_glob ; i++)
		delete def_glob[i] ;
	delete [] def_glob ;

	for (int i=0 ; i<ndef_glob ; i++)
		delete [] names_def_glob[i] ;
	delete [] names_def_glob ;
      
	for (int i=0 ; i<nopeuser ; i++)
	    delete names_opeuser[i] ;
	delete [] names_opeuser ;
	delete [] paruser ;

	for (int i=0 ; i<nopeuser_bin ; i++)
	    delete names_opeuser_bin[i] ;
	delete [] names_opeuser_bin ;
	delete [] paruser_bin ;

	if (name_met !=0x0)
		delete [] name_met ;

	for (int i=0 ; i<neq_int ; i++)
		delete eq_int[i] ;
	delete [] eq_int ;

	for (int i=0 ; i<neq ; i++)
		delete eq[i] ;
	delete [] eq ;

	for (int i=0 ; i<VARMAX ; i++)
		if (results[i]!=0x0)
			delete results[i] ;
	delete [] results ;

	if (which_coef!=0x0) {
		for (int i=0 ; i<nbr_conditions ; i++)
			delete which_coef[i] ;
		delete [] which_coef ;
	}
}

const Metric* System_of_eqs::get_met() const {
	if (met==0x0) {
		cerr << "Undefined metric in System_of_eqs" << endl ;
		abort() ;
	}
	return met ;
}

Term_eq* System_of_eqs::give_term_double (int which, int dd)  const {

	assert ((which>=0) && (which<nvar_double)) ;
	assert ((dd>=dom_min) && (dd<=dom_max)) ;

	int pos = which*ndom + (dd-dom_min) ;
	return term_double[pos] ;
}

Term_eq* System_of_eqs::give_term (int which, int dd)  const {

	assert ((which>=0) && (which<nvar)) ;
	assert ((dd>=dom_min) && (dd<=dom_max)) ;

	int pos = which*ndom + (dd-dom_min) ;
	return term[pos] ;
}

Term_eq* System_of_eqs::give_cst (int which, int dd) const {

	assert ((which>=0) && (which<ncst)) ;
	assert ((dd>=dom_min) && (dd<=dom_max)) ;
	int pos = which*ndom + (dd-dom_min) ;
	return cst[pos] ;
}

Term_eq* System_of_eqs::give_cst_hard (double xx, int dd) const {
	// Check if it has already been stored :
	for (int i=0 ; i<ncst_hard ; i++)
		if ((fabs(val_cst_hard(i) - xx) <PRECISION) && (cst_hard[i]->get_dom()==dd))
			return cst_hard[i] ;
	
	// Term not found, one needs to put it 
	cst_hard[ncst_hard] = new Term_eq (dd, xx) ;
	cst_hard[ncst_hard]->set_der_zero() ;
	val_cst_hard.set(ncst_hard) = xx ;
	ncst_hard++ ;
	return cst_hard[ncst_hard-1] ;
}

Ope_def* System_of_eqs::give_def (int which) const {

	assert ((which>=0) && (which<ndef)) ;
	return def[which] ;
}

Ope_def_global* System_of_eqs::give_def_glob (int which) const {

	assert ((which>=0) && (which<ndef_glob)) ;
	return def_glob[which] ;
}

void System_of_eqs::vars_to_terms () {

	for (int d=dom_min ; d<=dom_max ; d++)
	  espace.get_domain(d)->vars_to_terms() ;
  
	for (int vv=0 ; vv<nvar_double ; vv++)
		for (int dd=dom_min ; dd<=dom_max ; dd++)
			give_term_double(vv, dd)->set_val_d (*var_double[vv]) ;

	for (int vv=0 ; vv<nvar ; vv++)
		for (int dd=dom_min ; dd<=dom_max ; dd++) {
			espace.get_domain(dd)->zero_null_theta_mode (*var[vv], dd) ;
			give_term(vv, dd)->set_val_t (*var[vv]) ;
		}

}

// Add an uknown
void System_of_eqs::add_var (const char*nom, double& vv) {

	if ((nbr_char(nom, '_')!=0) || (nbr_char(nom, '^')!=0)) {
		cerr << "No indices allowed in names of variables" << endl ;
		abort() ;
	}
	
	var_double[nvar_double] = &vv ;
	names_var_double[nvar_double] = new char[LMAX];
	trim_spaces(names_var_double[nvar_double],nom) ;

	for (int dd=dom_min ; dd<=dom_max ; dd++) {
		term_double[nterm_double] = new Term_eq(dd, vv) ;
		assoc_var_double.set(nterm_double) = nvar_double ;
		nterm_double++ ;
	}
	nvar_double++ ;

	nbr_unknowns ++ ;
}

// Add an uknown
void System_of_eqs::add_var (const char*nom, Tensor& tt) {

	if (nom!=0x0) {
		if ((nbr_char(nom, '_')!=0) || (nbr_char(nom, '^')!=0)) {
			cerr << "No indices allowed in names of variables" << endl ;
			abort() ;
		}
	

	names_var[nvar] = new char[LMAX];
	trim_spaces(names_var[nvar],nom) ;
	}
 	else {
		names_var[nvar] = new char[LMAX] ;
		trim_spaces (names_var[nvar], "$") ;
	}

	assert (&tt.espace==&espace) ;
	var[nvar] = &tt ;
	for (int dd=dom_min ; dd<=dom_max ; dd++) {
		term[nterm] = new Term_eq(dd, tt) ;
		assoc_var.set(nterm) = nvar ;
		nterm++ ;
	}

	nvar++ ;
	
	for (int dd=dom_min ; dd<=dom_max ; dd++) 
	    nbr_unknowns += espace.get_domain(dd)->nbr_unknowns (tt, dd) ;
}

// Add a  constant
void System_of_eqs::add_cst (const char*nom, const Tensor& so) {

	if (nom!=0x0) {
	if ((nbr_char(nom, '_')!=0) || (nbr_char(nom, '^')!=0)) {
		cerr << "No indices allowed in names of constants" << endl ;
		abort() ;
	}

	char nomcst[LMAX];
	trim_spaces(nomcst,nom) ;

	names_cst[ncst] = new char[LMAX] ;
	trim_spaces (names_cst[ncst], nomcst) ;
	}
	else {
	  names_cst[ncst] = new char[LMAX] ;
	  trim_spaces (names_cst[ncst], "$") ;
	}
	
	ncst++ ;

	for (int dd=dom_min ; dd<=dom_max ; dd++) {
		cst[nterm_cst] = new Term_eq(dd, so) ;
		cst[nterm_cst]->set_der_zero() ;
		nterm_cst ++ ;
	}
}

void System_of_eqs::add_cst (const char*nom, double xx) {
	char  nomcst[LMAX];
	trim_spaces(nomcst,nom) ;

	names_cst[ncst] = new char[LMAX] ;
	trim_spaces (names_cst[ncst], nomcst) ;
	ncst++ ;

	for (int dd=dom_min ; dd<=dom_max ; dd++) {
		cst[nterm_cst] = new Term_eq(dd, xx) ;
		cst[nterm_cst]->set_der_zero() ;
		nterm_cst++ ;
	}
}

void System_of_eqs::add_ope (const char* nom, Term_eq (*pope) (const Term_eq&, Param*), Param* par) {
   
    names_opeuser[nopeuser] = new char [LMAX] ;
    trim_spaces (names_opeuser[nopeuser], nom) ;
    opeuser[nopeuser] = pope ;
    paruser[nopeuser] = par ;
    nopeuser++ ;
}

void System_of_eqs::add_ope (const char* nom, Term_eq (*pope) (const Term_eq&, const Term_eq&, Param*), Param* par) {

    names_opeuser_bin[nopeuser_bin] = new char [LMAX] ;
    trim_spaces (names_opeuser_bin[nopeuser_bin], nom) ;
    opeuser_bin[nopeuser_bin] = pope ;
    paruser_bin[nopeuser_bin] = par ;
    nopeuser_bin++ ;
}

// Add a  definition
void System_of_eqs::add_def (int dom, const char* nom) {

	// Récupère le = :
	char p1[LMAX] ;
	char p2[LMAX] ;
	bool indic = is_ope_bin(nom, p1, p2, '=') ;
	if (!indic) {
		cerr << "= needed for definitions" << endl ;
		abort() ;
	}

	//lhs is name of definition :
	names_def[ndef] = new char[LMAX] ;
	get_util (names_def[ndef], p1) ;
	
	int valence ;
	char* indices = 0x0 ;
	Array<int>* ttype = 0x0 ;
	bool auxi = is_tensor (p1, names_def[ndef], valence, indices, ttype) ;
	assert (auxi) ;
	
	def[ndef] = new Ope_def (this, give_ope (dom, p2), valence, indices, ttype) ;
	if (ttype!=0x0)
		delete ttype ;
	if (indices!=0x0)
		delete [] indices ;
	ndef++ ;
}

void System_of_eqs::add_def (const char* nom) {
  for (int dd=dom_min ; dd<=dom_max ; dd++)
    add_def(dd, nom) ;
}

// Add a definition invloving all the domains (i.e. integral)
void System_of_eqs::add_def_global (int dom, const char* nom) {

	// Récupère le = :
	char p1[LMAX] ;
	char p2[LMAX] ;
	bool indic = is_ope_bin(nom, p1, p2, '=') ;
	if (!indic) {
		cerr << "= needed for definitions" << endl ;
		abort() ;
	}
      
	//lhs is name of definition :
	names_def_glob[ndef_glob] = new char[LMAX] ;
	get_util (names_def_glob[ndef_glob], p1) ;
	def_glob[ndef_glob] = new Ope_def_global (this, dom, p2) ;
	ndef_glob++ ;
}

void System_of_eqs::add_def_global (const char* nom) {
  for (int dd=dom_min ; dd<=dom_max ; dd++)
    add_def_global(dd, nom) ;
}


void System_of_eqs::xx_to_ders (const Array<double>& xx) {

	assert (xx.get_ndim()==1) ;
	assert (xx.get_size(0) == nbr_unknowns) ;

	int conte = 0 ;
	int pos_term = 0 ;
	
	espace.xx_to_ders_variable_domains(xx, conte) ;
	
	for (int i=0 ; i<nvar_double ; i++) {
		for (int dd=dom_min ; dd<dom_max ; dd++) {
			term_double[pos_term]->set_der_d(xx(conte)) ;
			pos_term ++ ;
		}
		conte ++ ;
	}

	for (int i=0 ; i<nterm ; i++) {

		// Check for symetric tensor : 
		const Metric_tensor* pmet = dynamic_cast<const Metric_tensor*>(term[i]->val_t) ;
		if (pmet==0x0) {
			Tensor auxi (term[i]->get_val_t(), false) ;
	
			int dom = term[i]->get_dom() ;
			espace.get_domain(dom)->affecte_tau(auxi, dom, xx, conte) ;
			
			term[i]->set_der_t(auxi) ;
			}
		else  {
			Metric_tensor auxi (*pmet, false) ;
	
			int dom = term[i]->get_dom() ;
			espace.get_domain(dom)->affecte_tau(auxi, dom, xx, conte) ;
			term[i]->set_der_t(auxi) ;
		}
	}
}

void System_of_eqs::xx_to_vars (const Array<double>& xx, int& conte) {

	assert (xx.get_ndim()==1) ;
	assert (xx.get_size(0) == nbr_unknowns) ;
	
	for (int i=0 ; i<nvar_double ; i++) {
		*var_double[i] = xx(conte) ; 
		conte ++ ;
	}
	
	for (int i=0 ; i<nvar ; i++) 
	  for (int dom=dom_min ; dom<=dom_max ; dom++) {
		espace.get_domain(dom)->affecte_tau(*var[i], dom, xx, conte) ;
	}
	
}

Tensor System_of_eqs::give_val_def (const char* so) const {
  
  char name[LMAX] ;
  trim_spaces(name, so) ;
  
  bool found = false ;
  int valence = -1;
  Array<int>* index ;
  Base_tensor* basis ;
  
  for (int i=0 ; i<ndef ; i++) 
    if (!found) {
      if (strcmp(names_def[i], name)==0) {
	found = true ;
	Tensor auxi (give_def(i)->get_res()->get_val_t()) ;
	valence = auxi.get_valence() ;
	index = new Array<int> (auxi.get_index_type()) ;
	basis = new Base_tensor (auxi.get_basis()) ;
      }
    }
    
  if (!found) {
    std::string def(so);
    cerr << "Definition " << def << " not found" << endl ;
    abort() ;
  }
  
  Tensor res (espace, valence, *index, *basis) ;
  res = 0 ;
  
  delete index ;
  delete basis ;
  
  for (int i=0 ; i<ndef ; i++) 
    if (strcmp(names_def[i], name)==0) {
      int zedom = give_def(i)->get_res()->get_dom() ;
      Tensor auxi (give_def(i)->get_res()->get_val_t()) ;
      for (int n=0 ; n<res.get_n_comp() ; n++) {
	Array<int> ind (res.indices(n)) ;
	res.set(ind).set_domain(zedom) = auxi(ind)(zedom) ;
      }
	res.set_basis(zedom) = auxi.get_basis().get_basis(zedom) ;
    }

  return res ; 
}

}
