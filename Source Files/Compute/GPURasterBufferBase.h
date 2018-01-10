//Pure virtual function for the basis of all GPUBuffers

#pragma once

#include "Compute/GPUDevice.h"

#include "DeviceBuffer.h"
/*
*    Buffer for gpu/cpu communication and storage.
*    @tparam	T	Generic type parameter.
*/

template <class T>
class GPURasterBufferBase : public DeviceBuffer<T> {

public:

	/*
	*    Constructor.
	*    @param [in,out]	deviceIn		The device in.
	*    @param 		 	_objectCount	(Optional) Number of objects.
	*/

	GPURasterBufferBase(const GPUDevice& deviceIn, uint _size = 0)
		:DeviceBuffer(deviceIn, _size) {
		
	}

	/* Destructor. */
	virtual ~GPURasterBufferBase() {

	}

	/* Map resources. */
	virtual void MapResources() {

	}
	/* Unmap resources. */
	virtual void UnmapResources() {

	}


	/*
	*    Transfer to host.
	*    @param [in,out]	device	The device.
	*/

	virtual void TransferToHost() {

	}

	/*
	*    Transfer to device.
	*    @param [in,out]	device	The device.
	*/

	virtual void TransferToDevice() {

	}
	/*
	*    Bind data.
	*    @param	parameter1	The first parameter.
	*/

	virtual void BindData(uint) {

	}

};