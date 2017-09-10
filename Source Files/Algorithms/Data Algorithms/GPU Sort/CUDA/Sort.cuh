#pragma once

#include "Metrics.h"

namespace Sort {

	/*
	 *    Calculates.
	 *    @param 		 	size  	The size.
	 *    @param [in,out]	keys  	If non-null, the keys. Host pointer.
	 *    @param [in,out]	values	If non-null, the values. Host pointer.
	 *    @return	Null if it fails, else a pointer to a P.
	 */

	uint* Calculate(uint size, uint64* keys, uint* values);

}