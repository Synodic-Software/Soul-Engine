#pragma once

#include "GPGPU/GPUDevice.h"

/* Buffer for gpu. */

template <class T>
class GPUBuffer {

public:
	/* Default constructor. */
	GPUBuffer(GPUDevice& deviceIn, uint _byteCount) {
		byteCount = _byteCount;
	}
	/* Destructor. */
	virtual ~GPUBuffer(){}


	virtual void TransferToDevice(GPUDevice& device) {
		
	}

	virtual void TransferToHost(GPUDevice& device) {
		
	}

	/*
	 *    Gets the data.
	 *    @return	Null if it fails, else the data.
	 */

	void* Data() {
		return hostData;
	}

protected:
	/* The data */
	T* hostData;
	T* deviceData;

	uint byteCount;

private:

	//operator overloads
public:

	/*
	 *    T* casting operator.
	 *    @return	The device data. Facilitates passing in a GPUBuffer object to a kernal
	 */

	operator T*() const { return deviceData; }
	
};