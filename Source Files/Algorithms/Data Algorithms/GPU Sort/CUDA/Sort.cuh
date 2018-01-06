#pragma once

#include "GPGPU/GPUBuffer.h"
#include "Engine Core/Object/Face.h"

namespace Sort {

	/*
	 *    Calculates.
	 *    @param 		 	size  	The size.
	 *    @param [in,out]	keys  	If non-null, the keys. Host pointer.
	 *    @param [in,out]	values	If non-null, the values. Host pointer.
	 *    @return	Null if it fails, else a pointer to a P.
	 */

	void Sort(GPUBuffer<uint64_t>& keys, GPUBuffer<Face>& values);
}
