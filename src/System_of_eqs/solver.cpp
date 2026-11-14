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

#include <algorithm>
#include <chrono>
#include "system_of_eqs.hpp"
#include "scalar.hpp"
#include "metric.hpp"
namespace Kadath {
namespace {
/// Seconds since the first call; only used when System_of_eqs::profile_phases is set.
double phase_clock() {
	static auto const start = std::chrono::steady_clock::now() ;
	return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() ;
}
}
Array<double> System_of_eqs::check_equations() {

	
	Array<double> sec (sec_member()) ;
	Array<double> errors (neq_int + neq) ;
	errors = 0 ;
      
	int pos = 0 ;
	for (int i=0 ; i<neq_int ; i++) {
	  errors.set(i) = fabs(sec(pos)) ;
	  pos ++ ;
	}

	for (int i=0 ; i<neq ; i++) {
	    double max = 0 ;
	    for (int j=0 ; j<eq[i]->get_n_cond_tot() ; j++) {
	      if (fabs(sec(pos)) > max)
		max = fabs(sec(pos)) ;
	      pos ++ ;
	    }
	    errors.set(neq_int+i) = max ;
	}
      return errors ;
}

Array<double> System_of_eqs::sec_member() {

	der_clean.clear() ;

	vars_to_terms() ;

	if (met!=0x0)
	  for (int d=dom_min ; d<=dom_max ; d++)
		met->update(d) ;
	for (int i=0 ; i<ndef ; i++)
		def[i]->compute_res() ;

	int conte = 0 ;
	for (int i=0 ; i<neq ; i++)
	     eq[i]->apply(conte, results) ;

	// Need to assert the size :
	if (nbr_conditions==-1) {
		nbr_conditions = 0 ;
		for (int i=0 ; i<neq_int ; i++)
			nbr_conditions ++ ;
		for (int i=0 ; i<neq ; i++)
		  nbr_conditions += eq[i]->get_n_cond_tot() ;
	}

	// Computation of the second member itself :
	Array<double> res (nbr_conditions) ;
	res = 0 ;
	conte = 0 ;
	int pos_res = 0 ;
	for (int i=0 ; i<neq_int ; i++) {
		res.set(pos_res) = eq_int[i]->get_val() ;
		pos_res ++ ;
	}

	for (int i=0 ; i<neq ; i++)
	    eq[i]->export_val(conte, results, res, pos_res) ;
	return res ;
}

void System_of_eqs::find_variable_domains() {
	if (variable_domains_known)
		return ;
	variable_domains_known = true ;
	variable_domains.clear() ;
	int const nvd = espace.nbr_unknowns_from_variable_domains() ;
	for (int cc=0 ; cc<nvd ; cc++) {
		int conte = 0 ;
		Array<int> zedoms (2) ;
		zedoms = -1 ;
		espace.affecte_coef_to_variable_domains(conte, cc, zedoms) ;
		for (int d=0 ; d<zedoms.get_size(0) ; d++)
			if ((zedoms(d)!=-1) && (std::find(variable_domains.begin(), variable_domains.end(), zedoms(d))==variable_domains.end()))
				variable_domains.push_back(zedoms(d)) ;
	}
}

Array<double> System_of_eqs::do_JX (const Array<double>& xx) {

	double t_phase = profile_phases ? phase_clock() : 0. ;
	auto lap = [&](double& acc) {
		if (profile_phases) {
			double const now = phase_clock() ;
			acc += now - t_phase ;
			t_phase = now ;
		}
	} ;
	der_clean.clear() ;
	find_variable_domains() ;
	xx_to_ders(xx) ;
	// As in do_col_J (update_terms_from_variable_domains): in a variable domain the collocation points move with
	// the shape, so every term and constant field there also changes by update_term_eq's contribution, linear in
	// the domain derivative that xx_to_ders has just set. A term's derivative is the sum of both parts.
	if (!variable_domains.empty()) {
		auto is_variable = [this](int dom) {
			return std::find(variable_domains.begin(), variable_domains.end(), dom) != variable_domains.end() ;
		} ;
		for (int i=0 ; i<nterm ; i++)
			if (is_variable(term[i]->get_dom())) {
				Tensor const field_part (term[i]->get_der_t()) ;
				espace.get_domain(term[i]->get_dom())->update_term_eq(term[i]) ;
				term[i]->set_der_t(term[i]->get_der_t() + field_part) ;
			}
		for (int i=0 ; i<nterm_cst ; i++)
			if ((cst[i]->get_type_data()==TERM_T) && is_variable(cst[i]->get_dom()))
				espace.get_domain(cst[i]->get_dom())->update_term_eq(cst[i]) ;
			else
				cst[i]->set_der_zero() ;
	}
	lap(times_JX.setup) ;
	double const t_met = profile_phases ? phase_clock() : 0. ;
	if (met!=0x0)  
	  for (int d=dom_min ; d<=dom_max ; d++)
		met->update(d) ;
	if (profile_phases)
		times_JX.met += phase_clock() - t_met ;
	// Delete the definitions 
	for (int i=0 ; i<ndef ; i++)
		def[i]->compute_res() ;
	lap(times_JX.defs) ;

	int conte = 0 ;
	for (int i=0 ; i<neq ; i++)  
	    eq[i]->apply(conte, results) ;
	lap(times_JX.eqs) ;
	
	if (nbr_conditions==-1) {
		cerr << "Number of conditions unknown ; call sec_member first" << endl ;
		abort() ;
	}
	
	Array<double> res (nbr_conditions) ;
	conte = 0 ;
	int pos_res = 0 ;
	res = 0 ;
	for (int i=0 ; i<neq_int ; i++) {
		res.set(pos_res) = eq_int[i]->get_der() ;
		pos_res ++ ;
	}

	for (int i=0 ; i<neq ; i++)
	    eq[i]->export_der(conte, results, res, pos_res) ;
	lap(times_JX.exp) ;
	if (profile_phases)
		times_JX.calls ++ ;
	return res ;
}

Array<double> System_of_eqs::do_col_J (int cc) {

	assert ((cc>=0) && (cc<nbr_unknowns)) ;
	double t_phase = profile_phases ? phase_clock() : 0. ;
	auto lap = [&](double& acc) {
		if (profile_phases) {
			double const now = phase_clock() ;
			acc += now - t_phase ;
			t_phase = now ;
		}
	} ;

	// Affecte nterms derivatives :
	int conte = 0 ;
	int zedom = -1 ;
	bool is_var_double = false ;
	Array<int> zedoms (2) ;
	zedoms = - 1 ;
	// Variable Domains :
	espace.affecte_coef_to_variable_domains(conte, cc, zedoms) ;
	if (zedoms(0)!=-1)
	  update_terms_from_variable_domains(zedoms) ;
	else {
	  for (int i=0 ; i<nterm_cst ; i++)
	    cst[i]->set_der_zero() ;
	  for (int i=0 ; i<nterm ; i++)
	    term[i]->set_der_zero() ;
	}
	
	// Double 
	for (int i=0 ; i<nvar_double ; i++) {
		if (conte==cc) {
			for (int dd=dom_min ; dd<=dom_max ; dd++)
				term_double[i*ndom+(dd-dom_min)]->set_der_d(1.) ;
			is_var_double = true ;
		}
		else
			for (int dd=dom_min ; dd<=dom_max ; dd++)
				term_double[i*ndom+(dd-dom_min)]->set_der_d(0.) ;
		conte ++ ;
	}

	// Fields
	for (int i=0 ; i<nterm ; i++) {
		int dom = term[i]->get_dom() ;
		Tensor auxi (term[i]->get_val_t(), false) ;
		espace.get_domain(dom)->affecte_tau_one_coef(auxi, dom, cc, conte) ;
		for (int j=0 ; j<auxi.get_n_comp() ; j++) {
			// Si la base n'est pas affectee on la met
			if ((!auxi(auxi.indices(j))(dom).check_if_zero()) && (!auxi(auxi.indices(j))(dom).get_base().is_def()))
				auxi.set(auxi.indices(j)).set_domain(dom).set_base() = term[i]->get_val_t()(auxi.indices(j))(dom).get_base() ;
			if ((zedom==-1) && (!auxi(auxi.indices(j))(dom).check_if_zero()))
				zedom = dom ;
		}

		term[i]->set_der_t(auxi + term[i]->get_der_t()) ;
	}
	
	lap(times_col_J.setup) ;
	// Metric quantities and definitions: only where they can have changed. In a domain not touched by
	// this column every term has a zero derivative, so they keep their values with a zero derivative; if
	// that is what they already hold (der_clean) there is nothing to recompute.
	auto active = [&](int d) {
		return is_var_double || (d==zedom) || (d==zedoms(0)) || (d==zedoms(1)) ;
	} ;
	bool const track = skip_clean_domains && (ndef_glob==0) ;
	bool const skip = track && (static_cast<int>(der_clean.size())==ndom) ;
	auto todo = [&](int d) {
		return !skip || (d<dom_min) || (d>dom_max) || active(d) || !der_clean[d-dom_min] ;
	} ;
	double t_sub = profile_phases ? phase_clock() : 0. ;
	auto sub = [&](double& acc) {
		if (profile_phases) {
			double const now = phase_clock() ;
			acc += now - t_sub ;
			t_sub = now ;
		}
	} ;
	if (profile_phases && static_cast<int>(times_col_J.per_def.size())!=ndef)
		times_col_J.per_def.assign(ndef, 0.) ;
	if (met!=0x0)
	  for (int d=dom_min ; d<=dom_max ; d++)
		if (todo(d))
			met->update(d) ;
	sub(times_col_J.met) ;

	for (int i=0 ; i<ndef ; i++)
		if (todo(def[i]->get_dom())) {
			def[i]->compute_res() ;
			if (profile_phases)
				sub(times_col_J.per_def[i]) ;
		}

	if (track) {
		der_clean.assign(ndom, 1) ;
		for (int d=dom_min ; d<=dom_max ; d++)
			if (active(d))
				der_clean[d-dom_min] = 0 ;
	}
	else
		der_clean.clear() ;
	lap(times_col_J.defs) ;

	conte = 0 ;
	for (int i=0 ; i<neq ; i++) {
		if ((is_var_double) || (eq[i]->take_into_account(zedom)) || (eq[i]->take_into_account(zedoms(0))) || (eq[i]->take_into_account(zedoms(1)))) {
			eq[i]->apply(conte, results) ;
		}
		else
			conte += eq[i]->n_ope ;
	}
		
	lap(times_col_J.eqs) ;
	if (nbr_conditions==-1) {
		cerr << "Number of conditions unknown ; call sec_member first" << endl ;
		abort() ;
	}
	
	Array<double> res (nbr_conditions) ;
	conte = 0 ;
	int pos_res = 0 ;
	res = 0 ;
	for (int i=0 ; i<neq_int ; i++) {
		res.set(pos_res) = eq_int[i]->get_der() ;
		pos_res ++ ;
	} 
	
	for (int i=0 ; i<neq ; i++) {
		if ((is_var_double) || (eq[i]->take_into_account(zedom))|| (eq[i]->take_into_account(zedoms(0))) || (eq[i]->take_into_account(zedoms(1)))) {
		  eq[i]->export_der(conte, results, res, pos_res) ;
		}
		else
			{
			pos_res += eq[i]->n_cond_tot ;
			conte += eq[i]->n_ope ;
			}  
	}
	
	lap(times_col_J.exp) ;
	if (profile_phases)
		times_col_J.calls ++ ;
	return res ;
}

void System_of_eqs::update_terms_from_variable_domains(const Array<int>& zedoms) {
  
	for (int i=0 ; i<nterm ; i++) {
	      int dom = term[i]->get_dom() ;
	      bool  todo = false ;
	      for (int d=0 ; d<zedoms.get_size(0) ; d++)
		  if (zedoms(d)==dom)
		      todo = true ;
	      if (todo)
		espace.get_domain(dom)->update_term_eq (term[i]) ;
	      else
		term[i]->set_der_zero() ;
	}
	  
	  for (int i=0 ; i<nterm_cst ; i++) 
	      if (cst[i]->get_type_data() == TERM_T) {
	      int dom = cst[i]->get_dom() ;
	      bool  todo = false ;
	      for (int d=0 ; d<zedoms.get_size(0) ; d++)
		  if (zedoms(d)==dom)
		      todo = true ;
	      if (todo)
		espace.get_domain(dom)->update_term_eq (cst[i]) ;
	      else
		cst[i]->set_der_zero() ;
	  } 
}}
