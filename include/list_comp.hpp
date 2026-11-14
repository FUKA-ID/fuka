// clang-format off
/*
    Copyright 2018 Philippe Grandclement

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

#ifndef __LIST_COMP_HPP_
#define __LIST_COMP_HPP_

#include "headcpp.hpp"
#include "array.hpp"

namespace Kadath {
/**
* Class for storing a list of tensorial components.
*
* It simply consists of a list of integer Arrays. It is intended to be used when passing equations to a \c System_of_eqs
* \ingroup util
**/

class List_comp {
    protected:
        int ncomp ; ///< Number of stored components.
	int valence ; ///< Number of indices for each component.
	Array<int>  ** pcomp ; ///< Array of pointers on each componenent.

    public:
	/** Standard constructor
	* @param nc [input] number of components.
	* @param val [input] valence of the components.	
	**/
        List_comp (int nc, int val) ;
	List_comp (const List_comp&) ; ///< Constructor by copy.
	
	~List_comp() ; ///< Destructor
	
	/**
	* Read/write of one particular component.
	* @param i [input] which component.
	*/
	Array<int>* set(int i) ;

	/**
	* Read/write of one particular component.
	* @param i [input] which component.
	*/
	Array<int>* operator() (int i) const ;

	/**
	* Returns the number of components.
	*/
	int get_ncomp() const {return ncomp ;} ;

	/**
	* Returns a pointer of the liste
	*/
	Array<int>** get_pcomp() const {return pcomp ;} ;
	
} ;

}
#endif
