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

#ifndef __OPE_EQ_HPP_
#define __OPE_EQ_HPP_

#include "term_eq.hpp"
#include "memory.hpp"
#include <memory>

namespace Kadath {

/**
 * Abstract class that describes the various operators that can appear in the equations.
 * It can not be instanciated and one must use the derived classes.
 * It works at the \c Term_eq level (i.e. on a given \c Domain and using the dual quantities required by the automatic differentiation technique).
 * \ingroup systems
 */
class Ope_eq : public MemoryMappable {

	protected:
		const System_of_eqs* syst ; ///< The associated \c System_of_eqs
		int dom ; ///< Index of the \c Domain where the operator is defined.
		int n_ope ; ///< Number of terms involved (2 for + for instance, only one for sqrt...)
		Ope_eq** parts ; ///< Pointers of the various parts of the current operator.
		/**
		* Constructor. The various parts are uninitialized at this point.
		* @param syst : the associated \c System_of_eqs.
		* @param dom : the index of the \c Domain.
		* @param np : number of terms.
		*/
		Ope_eq (const System_of_eqs* syst, int dom, int np) ;
		/**
		* Constructor. The number of terms is undefined.
		* @param syst : the associated \c System_of_eqs.
		* @param dom : the index of the \c Domain.
		*/
		Ope_eq (const System_of_eqs* syst, int dom) ;
		Ope_eq (const Ope_eq&) ; ///< Copy constructor
	public:
		virtual ~Ope_eq() ; ///< Destructor

		/**
		* @return the index of the \c Domain.
		*/
		int get_dom() const {return dom; } ;

		/**
		* Computes the action of the current \c Ope_eq using its various parts.
		* In a Jacobian column (System_of_eqs::prune_active) a subtree whose derivative is zero is not
		* recomputed: its result (the current value with a zero derivative) is taken from a cache valid as long
		* as the fields are unchanged (System_of_eqs::field_generation).
		* @return the \c Term_eq containing the result.
		*/
		Term_eq action() const ;
		/**
		* Computes the action (implemented by each operator; called through \c action).
		*/
		virtual Term_eq compute() const = 0;
		/**
		* @return true if the derivative of the result is zero in the current Jacobian column, i.e. the operator
		* depends only on its parts (no metric, no other domain) and every part has a zero derivative.
		*/
		bool der_is_zero() const ;
	protected:
		/// Uncached test of \c der_is_zero (overridden by the leaves).
		virtual bool der_zero_local() const ;
	private:
		mutable long zero_stamp = -1 ; ///< Column stamp of \c zero_flag.
		mutable bool zero_flag = false ; ///< Result of \c der_is_zero for \c zero_stamp.
		mutable long cache_generation = -1 ; ///< Field generation of \c cache.
		mutable std::unique_ptr<Term_eq> cache ; ///< Result with a zero derivative.
		mutable bool der_request = false ; ///< Set by the parent: return the derivative (as the value) only.
		mutable long val_generation = -1 ; ///< Field generation of \c val_cache.
		mutable std::unique_ptr<Term_eq> val_cache ; ///< Value (no derivative) of the result.
		bool request_derivatives () const ; ///< Linear in the parts with a non-zero derivative: ask them for it.
		static Term_eq derivative_as_value (const Term_eq&) ; ///< The derivative of a result as the value of a \c Term_eq.
} ;

/**
 * The operator identity.
 * \ingroup systems.
 */
class Ope_id : public Ope_eq {

	protected:
		const Term_eq* target ; ///< The input \c Term_eq
		int valence ; ///< Valence of the result.
		char* name_ind ; ///< The names of the various indices (if a \c Tensor of valence >0)
		Array<int>* type_ind ; ///< The type of the indices.
		bool need_sum ; ///< True if an inner contraction is needed to compute the result.
		virtual bool der_zero_local() const ;

	public:
		/**
		* Constructor
		* @param syst : the associated \c System_of_eqs.
		* @param so : The input \c Term_eq
		* @param valence : valence of the result (can differ from the one of so, due to inner contraction).
		* @param names : name of the indices.
		* @param ttype : type of the indices (can differ from so, in whic case a \c Metric is required to do the manipulation).
		*/
		Ope_id (const System_of_eqs* syst, const Term_eq* so, int valence, char* names, Array<int>* ttype) ;
		/**
		* Constructor with mos of the stuff uninitialized.
		* @param syst : the associated \c System_of_eqs.
		* @param so : The input \c Term_eq
		*/
		Ope_id (const System_of_eqs*, const Term_eq*) ;
		virtual ~Ope_id() ; ///< Destructor.
	public:
		virtual Term_eq compute() const ;
} ;


/**
 * The operator power-law
 * \ingroup systems.
 */
class Ope_pow : public Ope_eq {

	protected:
		int power ; ///< The exponent (an integer, possibly negative).
	public:
		/**
		* Constructor
		* @param syst : the associated \c System_of_eqs.
		* @param pow : the exponent.
		* @param so : the source.
		*/
		Ope_pow(const System_of_eqs* syst, int pow, Ope_eq* so) ;
		virtual ~Ope_pow() ; ///< Destructor.

		virtual Term_eq compute() const ;
} ;

/**
 * The operator minus
 * \ingroup systems.
 */
class Ope_minus : public Ope_eq {

	public:
		/**
		* Constructor
		* @param syst : the associated \c System_of_eqs.
		* @param so : the source.
		*/
		Ope_minus(const System_of_eqs* syst, Ope_eq* so) ;
		virtual ~Ope_minus() ; ///< Destructor

		virtual Term_eq compute() const ;
} ;

/**
 * The operator addition
 * \ingroup systems.
 */
class Ope_add : public Ope_eq {

	public:
		/**
		* Constructor
		* @param syst : the associated \c System_of_eqs.
		* @param aa : first term.
		* @param bb : second term.
		*/
		Ope_add(const System_of_eqs*, Ope_eq* aa, Ope_eq* bb) ;
		virtual ~Ope_add() ; ///< Destructor

		virtual Term_eq compute() const ;
} ;

/**
 * The operator substraction
 * \ingroup systems.
 */
class Ope_sub : public Ope_eq {

	public:
		/**
		* Constructor
		* @param syst : the associated \c System_of_eqs.
		* @param aa : first term.
		* @param bb : second term.
		*/
		Ope_sub(const System_of_eqs* syst, Ope_eq* aa, Ope_eq* bb) ;
		virtual ~Ope_sub() ;///< Destructor

		virtual Term_eq compute() const ;
} ;

/**
 * The operator Multiplication.
 * When dealing with tensors it takes into account the possible contractions.
 * \ingroup systems.
 */
class Ope_mult : public Ope_eq {

	public:
		/**
		* Constructor
		* @param syst : the associated \c System_of_eqs.
		* @param aa : first term.
		* @param bb : second term.
		*/
		Ope_mult(const System_of_eqs* syst, Ope_eq* aa, Ope_eq* bb) ;
		virtual ~Ope_mult() ;///< Destructor

		virtual Term_eq compute() const ;
} ;

/**
 * The operator Division.
 * The second term must be a double or a \c Scalar
 * \ingroup systems.
 */
class Ope_div: public Ope_eq {

	public:
		/**
		* Constructor
		* @param syst : the associated \c System_of_eqs.
		* @param aa : first term.
		* @param bb : second term.
		*/
		Ope_div(const System_of_eqs* syst, Ope_eq* aa, Ope_eq* bb) ;
		virtual ~Ope_div() ;///< Destructor

		virtual Term_eq compute() const ;
} ;

/**
 * The operator Laplacian 3D.
 * Computes the flat 3D Laplacian
 * \ingroup systems.
 */
class Ope_lap: public Ope_eq {

	public:
		/**
		* Constructor
		* @param syst : the associated \c System_of_eqs.
		* @param so : the target.
		*/
		Ope_lap(const System_of_eqs* syst, Ope_eq* so) ;
		virtual ~Ope_lap() ; ///< Destructor

		virtual Term_eq compute() const ;
} ;

/**
 * The operator time derivative.
 * Computes the first time derivative
 * \ingroup systems.
 */
class Ope_dtime: public Ope_eq {

	public:
		/**
		* Constructor
		* @param syst : the associated \c System_of_eqs.
		* @param so : the target.
		*/
		Ope_dtime(const System_of_eqs* syst, Ope_eq* so) ;
		virtual ~Ope_dtime() ; ///< Destructor

		virtual Term_eq compute() const ;
} ;

/**
 * Second time derivative
 * Computes the second time derivative
 * \ingroup systems
 */

class Ope_ddtime: public Ope_eq {

	public:
		/**
		* Constructor
		* @param syst : the associated \c System_of_eqs
		* @param so : the target
		*/
		Ope_ddtime(const System_of_eqs* syst, Ope_eq* so) ;
		virtual ~Ope_ddtime() ; ///< Destructor

		virtual Term_eq compute() const ;
} ;

/**
 * The operator Laplacian 2D.
 * Computes the flat 2D Laplacian
 * \ingroup systems.
 */
class Ope_lap2: public Ope_eq {

	public:
		/**
		* Constructor
		* @param syst : the associated \c System_of_eqs.
		* @param so : the target.
		*/
		Ope_lap2(const System_of_eqs* syst, Ope_eq* so) ;
		virtual ~Ope_lap2() ;///< Destructor

		virtual Term_eq compute() const ;
} ;

/**
 * The operator normal derivative
 * Computes the derivative in the direction normal to a given boundary.
 * \ingroup systems.
 */
class Ope_dn: public Ope_eq {

	protected:
		int bound ; ///< The boundary

	public:
		/**
		* Constructor
		* @param syst : the associated \c System_of_eqs.
		* @param bb : name of the boundary.
		* @param so : the target.
		*/
		Ope_dn(const System_of_eqs* syst, int bb, Ope_eq* so) ;
		virtual ~Ope_dn() ;///< Destructor

		virtual Term_eq compute() const ;
} ;

/**
 * The operator flat gradient
 * Intended for systems where no metric has been defined.
 * \ingroup systems.
 */
class Ope_grad: public Ope_eq {

	public:
		/**
		* Constructor
		* @param syst : the associated \c System_of_eqs.
		* @param so : the target.
		*/
		Ope_grad(const System_of_eqs* syst, Ope_eq* so) ;
		virtual ~Ope_grad() ; ///< Destructor

		virtual Term_eq compute() const ;
} ;

/**
 * The operator flat scalar product
 * Intended for systems where no metric has been defined.
 * \ingroup systems.
 */
class Ope_scal: public Ope_eq {

	public:
		/**
		* Constructor
		* @param syst : the associated \c System_of_eqs.
		* @param aa : first vector.
		* @param bb : second vector.
		*/
		Ope_scal(const System_of_eqs* syst, Ope_eq* aa, Ope_eq* bb) ;
		virtual ~Ope_scal() ; ///< Destructor

		virtual Term_eq compute() const ;
} ;

/**
 * The operator covariant derivative.
 * Inner summation is performed, if need be.
 * \ingroup systems.
 */
class Ope_der: public Ope_eq {

	protected:
		int type_der ; ///< Type of derivative (CON or COV)
		char ind_der ; ///< Name of the index of the derivative.

	public:
		/**
		* Constructor
		* @param syst : the associated \c System_of_eqs.
		* @param typeder : type of derivative (CON or COV)
		* @param indder : name of the index of the derivative.
		* @param so : target
		*/
		Ope_der(const System_of_eqs* syst, int typeder, char indder, Ope_eq* so) ;
		virtual ~Ope_der() ; ///< Destructor

		virtual Term_eq compute() const ;
} ;

/**
 * The operator covariant derivative with respect to the flat metric.
 * Inner summation is performed, if need be.
 * \ingroup systems.
 */
class Ope_der_flat: public Ope_eq {

	protected:
		int type_der ;///< Type of derivative (CON or COV)
		char ind_der ;///< Name of the index of the derivative.

	public:
		/**
		* Constructor
		* @param syst : the associated \c System_of_eqs.
		* @param typeder : type of derivative (CON or COV)
		* @param indder : name of the index of the derivative.
		* @param so : target
		*/
		Ope_der_flat(const System_of_eqs* syst, int typeder, char indder, Ope_eq* so) ;
		virtual ~Ope_der_flat() ; ///< Destructor

		virtual Term_eq compute() const ;
} ;

/**
 * The operator covariant derivative with respect to the background metric.
 * Inner summation is performed, if need be.
 * \ingroup systems.
 */
class Ope_der_background: public Ope_eq {

	protected:
		int type_der ;///< Type of derivative (CON or COV)
		char ind_der ;///< Name of the index of the derivative.

	public:
		/**
		* Constructor
		* @param syst : the associated \c System_of_eqs.
		* @param typeder : type of derivative (CON or COV)
		* @param indder : name of the index of the derivative.
		* @param so : target
		*/
		Ope_der_background(const System_of_eqs* syst, int typeder, char indder, Ope_eq* so) ;
		virtual ~Ope_der_background() ; ///< Destructor

		virtual Term_eq compute() const ;
} ;

/**
 * The operator surface integral.
 * \ingroup systems.
 */
class Ope_int: public Ope_eq {

	protected:
		int bound ; ///< Boundary where the integral is computed.

	public:
		/**
		* Constructor
		* @param syst : the associated \c System_of_eqs.
		* @param bb : the boundary
		* @param so : target
		*/
		Ope_int(const System_of_eqs* syst, int bb, Ope_eq* so) ;
		virtual ~Ope_int() ; ///< Destructor

		virtual Term_eq compute() const ;
} ;

/**
 * The operator volume integral (in a given \c Domain)
 * \ingroup systems.
 */
class Ope_int_volume: public Ope_eq {

	public:
		/**
		* Constructor
		* @param syst : the associated \c System_of_eqs.
		* @param so : target
		*/
		Ope_int_volume (const System_of_eqs* syst, Ope_eq* so) ;
		virtual ~Ope_int_volume() ; ///< Destructor

		virtual Term_eq compute() const ;
} ;

/**
 * The operator definition.
 * It corresponds to expressions defined by the user in the\c System_of_eqs.
 * Indices may have to be renamed and/or summed, depending on their name.
 * \ingroup systems.
 */
class Ope_def: public Ope_eq {
	protected:
		Term_eq* res ; ///< Result of the current definition.
	public:
		/**
		* Constructor
		* @param syst : the associated \c System_of_eqs.
		* @param so : target
		* @param val : valence of the result (can be different from so, due to summations)
		* @param name : names of the indices
		* @param ttype : type of the various indices (COV or CON).
		*/
		Ope_def (const System_of_eqs* syst, Ope_eq* so, int val, char* name, Array<int>* ttype) ;
		virtual ~Ope_def() ;
		virtual Term_eq compute() const ;
		Term_eq* get_res() ; ///< Returns the result.
		void compute_res() ; ///< Forces the computation of the result (when things have changed).
} ;

/**
 * The operator multiplication by \f$r\f$.
 * \ingroup systems.
 */
class Ope_mult_r: public Ope_eq {

	public:
		/**
		* Constructor
		* @param syst : the associated \c System_of_eqs.
		* @param so : target
		*/
		Ope_mult_r (const System_of_eqs* syst, Ope_eq* so) ;
		virtual ~Ope_mult_r() ; ///< Destructor.

		virtual Term_eq compute() const ;
} ;

/**
 * The operator multiplication by \f$x\f$ (what it means depend on the \c Space considered).
 * \ingroup systems.
 */
class Ope_mult_x: public Ope_eq {

	public:
		/**
		* Constructor
		* @param syst : the associated \c System_of_eqs.
		* @param so : target
		*/
		Ope_mult_x (const System_of_eqs* syst, Ope_eq* so) ;
		virtual ~Ope_mult_x() ; ///< Destructor

		virtual Term_eq compute() const ;
} ;

/**
 * The operator multiplication by \f$\frac{1}{r} \partial_r\f$.
 * \ingroup systems.
 */
class Ope_srdr: public Ope_eq {

	public:
		/**
		* Constructor
		* @param syst : the associated \c System_of_eqs.
		* @param so : target
		*/
		Ope_srdr (const System_of_eqs* syst, Ope_eq* so) ;
		virtual ~Ope_srdr() ; ///< Destructor

		virtual Term_eq compute() const ;
} ;



/**
 * The operator second radial derivative
 * \ingroup systems.
 */
class Ope_ddr: public Ope_eq {

	public:
		/**
		* Constructor
		* @param syst : the associated \c System_of_eqs.
		* @param so : target
		*/
		Ope_ddr (const System_of_eqs* syst, Ope_eq* so) ;
		virtual ~Ope_ddr() ; ///< Destructor

		virtual Term_eq compute() const ;
} ;

/**
 * The operator first radial derivative
 * \ingroup systems.
 */
class Ope_dr: public Ope_eq {

	public:
		/**
		* Constructor
		* @param syst : the associated \c System_of_eqs.
		* @param so : target
		*/
		Ope_dr (const System_of_eqs* syst, Ope_eq* so) ;
		virtual ~Ope_dr() ; ///< Destructor

		virtual Term_eq compute() const ;
} ;

/**
 * The operator second derivative wrt \f$\varphi\f$.
 * \ingroup systems.
 */
class Ope_ddp: public Ope_eq {

	public:
		/**
		* Constructor
		* @param syst : the associated \c System_of_eqs.
		* @param so : target
		*/
		Ope_ddp (const System_of_eqs* syst, Ope_eq* so) ;
		virtual ~Ope_ddp() ; ///< Destructor

		virtual Term_eq compute() const ;
} ;

/**
 * The operator first derivative wrt \f$\theta\f$.
 * \ingroup systems.
 */
class Ope_dt: public Ope_eq {

	public:
		/**
		* Constructor
		* @param syst : the associated \c System_of_eqs.
		* @param so : target
		*/
		Ope_dt (const System_of_eqs* syst, Ope_eq* so) ;
		virtual ~Ope_dt() ; ///< Destructor

		virtual Term_eq compute() const ;
} ;

/**
 * The operator second derivative wrt \f$\theta\f$.
 * \ingroup systems.
 */
class Ope_ddt: public Ope_eq {

	public:
		/**
		* Constructor
		* @param syst : the associated \c System_of_eqs.
		* @param so : target
		*/
		Ope_ddt (const System_of_eqs* syst, Ope_eq* so) ;
		virtual ~Ope_ddt() ; ///< Destructor

		virtual Term_eq compute() const ;
} ;

/**
 * The operator division by \f$rf$.
 * \ingroup systems.
 */
class Ope_div_r: public Ope_eq {

	public:
		/**
		* Constructor
		* @param syst : the associated \c System_of_eqs.
		* @param so : target
		*/
		Ope_div_r (const System_of_eqs* syst, Ope_eq* so) ;
		virtual ~Ope_div_r() ; ///< Destructor

		virtual Term_eq compute() const ;
} ;

/**
 * The operator division by \f$r\sin\theta\f$.
 * \ingroup systems.
 */
class Ope_div_rsint: public Ope_eq {

	public:
		/**
		* Constructor
		* @param syst : the associated \c System_of_eqs.
		* @param so : target
		*/
		Ope_div_rsint (const System_of_eqs* syst, Ope_eq* so) ;
		virtual ~Ope_div_rsint() ; ///< Destructor

		virtual Term_eq compute() const ;
} ;

/**
 * The operator multiplication by \f$r\sin\theta\f$.
 * \ingroup systems.
 */
class Ope_mult_rsint: public Ope_eq {

	public:
		/**
		* Constructor
		* @param syst : the associated \c System_of_eqs.
		* @param so : target
		*/
		Ope_mult_rsint (const System_of_eqs* syst, Ope_eq* so) ;
		virtual ~Ope_mult_rsint() ; ///< Destructor

		virtual Term_eq compute() const ;
} ;

/**
 * The operator division by \f$r\cos\theta\f$.
 * \ingroup systems.
 */
class Ope_div_cost: public Ope_eq {

	public:
		/**
		* Constructor
		* @param syst : the associated \c System_of_eqs.
		* @param so : target
		*/
		Ope_div_cost (const System_of_eqs* syst, Ope_eq* so) ;
		virtual ~Ope_div_cost() ; ///< Destructor

		virtual Term_eq compute() const ;
} ;



/**
 * The operator division by \f$x+1\f$.
 * \ingroup systems.
 */
class Ope_div_xpone: public Ope_eq {

	public:
		/**
		* Constructor
		* @param syst : the associated \c System_of_eqs.
		* @param so : target
		*/
		Ope_div_xpone (const System_of_eqs* syst, Ope_eq* so) ;
		virtual ~Ope_div_xpone() ; ///< Destructor

		virtual Term_eq compute() const ;
} ;

/**
 * The operator division by \f$1-x^2\f$.
 * \ingroup systems.
 */
class Ope_div_1mx2: public Ope_eq {

	public:
		/**
		* Constructor
		* @param syst : the associated \c System_of_eqs.
		* @param so : target
		*/
		Ope_div_1mx2 (const System_of_eqs* syst, Ope_eq* so) ;
		virtual ~Ope_div_1mx2() ; ///< Destructor

		virtual Term_eq compute() const ;
} ;

/**
 * The operator division by \f$1-r/L\f$ (for AADS spacetimes).
 * \ingroup systems.
 */
class Ope_div_1mrsL: public Ope_eq {

	public:
		/**
		* Constructor
		* @param syst : the associated \c System_of_eqs.
		* @param so : target
		*/
		Ope_div_1mrsL (const System_of_eqs* syst, Ope_eq* so) ;
		virtual ~Ope_div_1mrsL() ; ///< Destructor

		virtual Term_eq compute() const ;
} ;

/**
 * The operator multiplication by \f$1-r/L\f$ (for AADS spacetimes).
 * \ingroup systems.
 */
class Ope_mult_1mrsL: public Ope_eq {

	public:
		/**
		* Constructor
		* @param syst : the associated \c System_of_eqs.
		* @param so : target
		*/
		Ope_mult_1mrsL (const System_of_eqs* syst, Ope_eq* so) ;
		virtual ~Ope_mult_1mrsL() ; ///< Destructor

		virtual Term_eq compute() const ;
} ;

/**
 * The operator multiplication by \f$\sin\theta\f$.
 * \ingroup systems.
 */
class Ope_mult_sint: public Ope_eq {

	public:
		/**
		* Constructor
		* @param syst : the associated \c System_of_eqs.
		* @param so : target
		*/
		Ope_mult_sint (const System_of_eqs* syst, Ope_eq* so) ;
		virtual ~Ope_mult_sint() ; ///> Destructor

		virtual Term_eq compute() const ;
} ;

/**
 * The operator division by \f$\sin\theta\f$.
 * \ingroup systems.
 */
class Ope_div_sint: public Ope_eq {

	public:
		/**
		* Constructor
		* @param syst : the associated \c System_of_eqs.
		* @param so : target
		*/
		Ope_div_sint (const System_of_eqs* syst, Ope_eq* so) ;
		virtual ~Ope_div_sint() ; ///> Destructor

		virtual Term_eq compute() const ;
} ;


/**
 * The operator partial derivative
 * \ingroup systems.
 */
class Ope_partial : public Ope_eq {
	protected:
		char ind_der ; ///< name of the index
	public:
		/**
		* Constructor
		* @param syst : the associated \c System_of_eqs.
		* @param name : name of the index corresponding to the derivative
		* @param so : target
		*/
		Ope_partial (const System_of_eqs* syst, char name, Ope_eq* so) ;
		virtual ~Ope_partial() ; ///< Destructor

		virtual Term_eq compute() const ;
} ;

/**
 * The operator determinant
 * \ingroup systems.
 */
class Ope_determinant : public Ope_eq {
	public:
		/**
		* Constructor
		* @param syst : the associated \c System_of_eqs.
		* @param so : target
		*/
		Ope_determinant (const System_of_eqs* syst, Ope_eq* so) ;
		virtual ~Ope_determinant() ; ///< Destructor

		virtual Term_eq compute() const ;
} ;

/**
 * The operator inverse (of a \c Metric_tensor ; i.e. rank 2 symmetric tensor).
 * \ingroup systems.
 */
class Ope_inverse : public Ope_eq {
	public:
		/**
		* Constructor
		* @param syst : the associated \c System_of_eqs.
		* @param so : target
		*/
		Ope_inverse (const System_of_eqs* syst, Ope_eq* so) ;
		virtual ~Ope_inverse() ; ///< Destructor

		virtual Term_eq compute() const ;
} ;

/**
 * The operator inverse (of a \c Metric_tensor ; i.e. rank 2 symmetric tensor).
 * It does not compute the true inverse in the sens that the cofactors are not divided by the determinant.
 * \ingroup systems.
 */
class Ope_inverse_nodet : public Ope_eq {
	public:
		/**
		* Constructor
		* @param syst : the associated \c System_of_eqs.
		* @param so : target
		*/
		Ope_inverse_nodet (const System_of_eqs* syst, Ope_eq* so) ;
		virtual ~Ope_inverse_nodet() ; ///< Destructor

		virtual Term_eq compute() const ;
} ;

/**
 * The operator partial derivative wrt one variable (same thing as Ope_partial ??)
 * \ingroup systems.
 */
class Ope_partial_var : public Ope_eq {
	protected:
	  int which_var ;

	public:
		/**
		* Constructor
		* @param syst : the associated \c System_of_eqs.
		* @param so : target
		*/
		Ope_partial_var (const System_of_eqs* syst, int, Ope_eq* so) ;
		virtual ~Ope_partial_var() ; ///< Destructor

		virtual Term_eq compute() const ;
} ;

/**
 * This operator gives the value of one coefficient of a field, on a given boundary.
 * \ingroup systems.
 */
class Ope_mode : public Ope_eq {
	protected:
		int bound ; ///< The boundary where the coefficients are read.
		/**
		* The desired coefficient.
		* The index corresponding to the boundary is unused.
		*/
		Index pos_cf ;
		double value ; ///< The result is the coefficient minus value.

	public:
		/**
		* Constructor
		* @param syst : the associated \c System_of_eqs.
		* @param bb : the boundary
		* @param ind : which coefficient.
		* @param val : the value with which the coefficient is "compared"
		* @param so : target
		*/
		Ope_mode (const System_of_eqs*, int bb, const Index& ind, double val, Ope_eq* so) ;
		virtual ~Ope_mode() ; ///< Destructor

		virtual Term_eq compute() const ;
} ;

/**
 * This operator gives the value of one coefficient of a field.
 * \ingroup systems.
 */
class Ope_val_mode : public Ope_eq {
	protected:
		Index pos_cf ; ///< The desired coefficient.
		double value ;///< The result is the coefficient minus value.

	public:
		/**
		* Constructor
		* @param syst : the associated \c System_of_eqs.
		* @param ind : which coefficient.
		* @param val : the value with which the coefficient is "compared"
		* @param so : target
		*/
		Ope_val_mode (const System_of_eqs* syst, const Index& ind, double val, Ope_eq* so) ;
		virtual ~Ope_val_mode() ; ///< Destructor

		virtual Term_eq compute() const ;
} ;

/**
 * This operator gives the value of a field at a given collocation point.
 * \ingroup systems.
 */
class Ope_val : public Ope_eq {
	protected:
		Index pos ; ///< which collocation point.

	public:
		/**
		* Constructor
		* @param syst : the associated \c System_of_eqs.
		* @param ind : which  collocation point.
		* @param so : target
		*/
		Ope_val(const System_of_eqs* syst, const Index& ind, Ope_eq* so) ;
		virtual ~Ope_val() ; ///< Destructor

		virtual Term_eq compute() const ;
} ;

/**
 * This operator gives the value of a field at a point (arbitrary not necesseraly a collocation one)
 * \ingroup systems.
 */
class Ope_point : public Ope_eq {
	protected:
		Point num; ///< Absolute coordinates of the point

	public:
		/**
		* Constructor
		* @param syst : the associated \c System_of_eqs.
		* @param pp : which point.
		* @param so : target
		*/
		Ope_point(const System_of_eqs* syst, const Point& pp, Ope_eq* so) ;
		virtual ~Ope_point() ; ///< Destructor

		virtual Term_eq compute() const ;
} ;

/**
 * This operator gives the value of a field at the origin
 * \ingroup systems.
 */
class Ope_val_ori : public Ope_eq {

	public:
		/**
		* Constructor
		* @param syst : the associated \c System_of_eqs.
		* @param dd : index of the \c Domain where the origin is (could be different from 0).
		* @param so : target
		*/
		Ope_val_ori(const System_of_eqs* syst, int dd, Ope_eq* so) ;
		virtual ~Ope_val_ori() ; ///< Destructor

		virtual Term_eq compute() const ;
} ;

/**
 * Operator square-root (only defined for a scalar field or a double)
 * \ingroup systems.
 */
class Ope_sqrt: public Ope_eq {

	public:
		/**
		* Constructor
		* @param syst : the associated \c System_of_eqs.
		* @param so : target
		*/
		Ope_sqrt(const System_of_eqs* syst, Ope_eq* so) ;
		virtual ~Ope_sqrt() ; ///< Destructor

		virtual Term_eq compute() const ;
} ;

/**
 * Operator square-root (only defined for a scalar field or a double).
 * Version where the output base is not standard but the anti standard base
 */
class Ope_sqrt_anti: public Ope_eq {

	public:
		/**
		* Constructor
		* @param syst : the associated \c System_of_eqs.
		* @param so : target
		*/
		Ope_sqrt_anti(const System_of_eqs* syst, Ope_eq* so) ;
		virtual ~Ope_sqrt_anti() ; ///< Destructor

		virtual Term_eq compute() const ;
} ;

/**
 * Operator square-root (only defined for a scalar field or a double).
 * Version where the output base is not standard but the one of rho (polar coordinates).
 * \ingroup systems.
 */
class Ope_sqrt_nonstd: public Ope_eq {

	public:
		/**
		* Constructor
		* @param syst : the associated \c System_of_eqs.
		* @param so : target
		*/
		Ope_sqrt_nonstd(const System_of_eqs* syst, Ope_eq* so) ;
		virtual ~Ope_sqrt_nonstd() ; ///< Destructor

		virtual Term_eq compute() const ;
} ;

/**
 * Operator exponential (only defined for a scalar field or a double)
 * \ingroup systems.
 */
class Ope_exp: public Ope_eq {

	public:
		/**
		* Constructor
		* @param syst : the associated \c System_of_eqs.
		* @param so : target
		*/
		Ope_exp(const System_of_eqs* syst, Ope_eq* so) ;
		virtual ~Ope_exp() ; ///< Destructor

		virtual Term_eq compute() const ;
} ;

/**
 * Operator logarithm (only defined for a scalar field or a double)
 * \ingroup systems.
 */
class Ope_log: public Ope_eq {

	public:
		/**
		* Constructor
		* @param syst : the associated \c System_of_eqs.
		* @param so : target
		*/
		Ope_log(const System_of_eqs* syst, Ope_eq* so) ;
		virtual ~Ope_log() ; ///< Destructor.

		virtual Term_eq compute() const ;
} ;

/**
 * Operator  hyperbolic cosine (only defined for a scalar field or a double)
 * \ingroup systems.
 */
class Ope_cosh: public Ope_eq {

	public:
		/**
		* Constructor
		* @param syst : the associated \c System_of_eqs.
		* @param so : target
		*/
		Ope_cosh(const System_of_eqs* syst, Ope_eq* so) ;
		virtual ~Ope_cosh() ; ///< Destructor.

		virtual Term_eq compute() const ;
} ;

/**
 * Operator  hyperbolic sine (only defined for a scalar field or a double)
 * \ingroup systems.
 */
class Ope_sinh: public Ope_eq {

	public:
		/**
		* Constructor
		* @param syst : the associated \c System_of_eqs.
		* @param so : target
		*/
		Ope_sinh(const System_of_eqs* syst, Ope_eq* so) ;
		virtual ~Ope_sinh() ; ///< Destructor.

		virtual Term_eq compute() const ;
} ;


/**
 * Operator inverse hyperbolic tangent (only defined for a scalar field or a double)
 * \ingroup systems.
 */
class Ope_atanh: public Ope_eq {

	public:
		/**
		* Constructor
		* @param syst : the associated \c System_of_eqs.
		* @param so : target
		*/
		Ope_atanh(const System_of_eqs* syst, Ope_eq* so) ;
		virtual ~Ope_atanh() ; ///< Destructor.

		virtual Term_eq compute() const ;
} ;

/**
 * Operator cosine (only defined for a scalar field or a double)
 * \ingroup systems.
 */
class Ope_cos: public Ope_eq {

	public:
		/**
		* Constructor
		* @param syst : the associated \c System_of_eqs.
		* @param so : target
		*/
		Ope_cos(const System_of_eqs* syst, Ope_eq* so) ;
		virtual ~Ope_cos() ; ///< Destructor.

		virtual Term_eq compute() const ;
} ;

/**
 * Operator sine (only defined for a scalar field or a double)
 * \ingroup systems.
 */
class Ope_sin: public Ope_eq {

	public:
		/**
		* Constructor
		* @param syst : the associated \c System_of_eqs.
		* @param so : target
		*/
		Ope_sin(const System_of_eqs* syst, Ope_eq* so) ;
		virtual ~Ope_sin() ; ///< Destructor.

		virtual Term_eq compute() const ;
} ;

/**
 * Operator that fits a field to outgoing waves (highly specialized stuff)
 * \ingroup systems.
 */
class Ope_fit_waves : public Ope_eq {
	public:
		/**
		* Constructor
		* @param syst : the associated \c System_of_eqs.
		* @param so : target field
		* @param ome : the orbital frequency (assumes an helical Killing vector).
		*/
	    Ope_fit_waves (const System_of_eqs* syst, Ope_eq* so, Ope_eq* ome) ;
	    virtual ~Ope_fit_waves() ; ///< Destructor

	    virtual Term_eq compute() const ;
} ;

/**
 * Operator defined by the user in the \c System_of_eqs
 * This version is intended to work with one argument.
 * \ingroup systems.
 */
class Ope_user: public Ope_eq {
	protected:
		Param* par ; ///< Parameters required by the function.
		Term_eq (*pope) (const Term_eq&, Param*) ; ///< The function that implements the action of the operator.

	public:
		/**
		* Constructor
		* @param syst : the associated \c System_of_eqs.
		* @param zeope : pointer on the function that implements the action of the operator.
		* @param par : parameters required by the function.
		* @param so : the argument.
		*/
		Ope_user (const System_of_eqs* syst, Term_eq (*zeope) (const Term_eq&, Param*), Param* par, Ope_eq* so) ;
		virtual ~Ope_user() ; ///< Destructor
		virtual Term_eq compute() const ;
} ;

/**
 * Operator defined by the user in the \c System_of_eqs
 * This version is intended to work with two arguments.
 * \ingroup systems.
 */
class Ope_user_bin: public Ope_eq {
	protected:
		Param* par ;  ///< Parameters required by the function.
		Term_eq (*pope) (const Term_eq&, const Term_eq&, Param*) ;  ///< The function that implements the action of the operator.

	public:
		/**
		* Constructor
		* @param syst : the associated \c System_of_eqs.
		* @param zeope : pointer on the function that implements the action of the operator.
		* @param par : parameters required by the function.
		* @param aa : the first argument.
		* @param bb : the second argument.
		*/
		Ope_user_bin (const System_of_eqs* syst, Term_eq (*zeope) (const Term_eq&, const Term_eq&, Param*), Param* par, Ope_eq* aa, Ope_eq* bb) ;
		virtual ~Ope_user_bin() ; ///< Destructor.
		virtual Term_eq compute() const ;
} ;


/**
 * Operator importing the values of a field from a neighborig \c Domain
 * \ingroup systems.
 */
class Ope_import: public Ope_eq {

	protected:
		int bound ; ///< The boundary where the field is imported.
		/**
		 * 2d array containing.
		 * \li in (0,*) the indexes of the domains situated on the other side of the boundary.
		 * \li in (1,*) the name of the boundary, as seen by the other domains.
		*/
		Array<int> others ;

	public:
		/**
		* Constructor
		* @param syst : the associated \c System_of_eqs.
		* @param dd : index of the domain considered.
		* @param bb : the boundary.
		* @param field : the quantity to be imported.
		*/
		Ope_import(const System_of_eqs* syst, int dd, int bb, const char* field) ;
		virtual ~Ope_import() ; ///< Destructor

		virtual Term_eq compute() const ;
} ;

/**
 * Operator changin the tensorial basis of a field.
 * \ingroup systems.
 */
class Ope_change_basis: public Ope_eq {

	protected:
		int target_basis ; ///< The desired tensorial basis.

	public:
		/**
		* Constructor
		* @param syst : the associated \c System_of_eqs.
		* @param target : the tensorial basis of the result.
		* @param so : the target field.
		*/
		Ope_change_basis(const System_of_eqs* syst, int target, Ope_eq* so) ;
		virtual ~Ope_change_basis() ; ///< Destructor

		virtual Term_eq compute() const ;
} ;

/**
 * Operator for a global definition (i.e. involving all the domains ; line an integral in the whole space).
 * The value is a \c Term_eq and so defined in a given domain (even if its value is computed from all the domains...)
 * \ingroup systems
 */
class Ope_def_global : public Ope_eq {
	protected:
		Term_eq* res ; ///< Result
		Term_eq** auxi ; ///< Various parts of the result (i.e. the contributions of the various domains).
	public:
		/**
		* Constructor
		* @param syst : the associated \c System_of_eqs.
		* @param dom : the index of the \c Domain of the result.
		* @param name_ope : the quantity (typically should contain things like integvolume)
		*/
		Ope_def_global (const System_of_eqs* syst, int dom, const char* name_ope) ;
		virtual ~Ope_def_global() ;
		virtual Term_eq compute() const ;
		Term_eq* get_res() ;///< Returns the result.
		void compute_res() ; ///< Forces the computation of the result (when things have changed).
} ;
}
#endif
