#pragma once

#include "Compute/ComputeBuffer.h"
#include "Data/Structures/Geometry/Face.h"

namespace Sort {

	/*
	 *    Calculates.
	 *    @param 		 	size  	The size.
	 *    @param [in,out]	keys  	If non-null, the keys. Host pointer.
	 *    @param [in,out]	values	If non-null, the values. Host pointer.
	 *    @return	Null if it fails, else a pointer to a P.
	 */

	void Sort(ComputeBuffer<uint64_t>& keys, ComputeBuffer<Face>& values);
}
