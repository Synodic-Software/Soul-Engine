//The wrapper which passes all functions from user to actual buffer

#pragma once

#include "GPGPU/GPUDevice.h"
#include "GPGPU/GPUBufferBase.h"

#include "GPGPU/CUDA/CUDABuffer.h"
#include "GPGPU/OpenCL/OpenCLBuffer.h"

/*
 *    Buffer for gpu/cpu communication and storage.
 *    @tparam	T	Generic type parameter.
 */

template <class T>
class GPUBuffer {

public:

	/*
	 *    Constructor.
	 *    @param [in,out]	deviceIn		The device in.
	 *    @param 		 	_objectCount	(Optional) Number of objects.
	 */

	GPUBuffer(const GPUDevice& deviceIn, uint _size = 0) {

		if(deviceIn.api==CUDA) {
			buffer = new CUDABuffer<T>(deviceIn, _size);
		}
		else if(deviceIn.api == OpenCL) {
			buffer = new OpenCLBuffer<T>(deviceIn, _size);
		}

	}

	/* Destructor. */
	~GPUBuffer() {
		delete buffer;
	}

	/*
	 *    Transfer to host.
	 *    @param [in,out]	device	The device.
	 */

	void TransferToHost() const {
		buffer->TransferToHost();
	}

	/*
	 *    Transfer to device.
	 *    @param [in,out]	device	The device.
	 */

	void TransferToDevice() const {
		buffer->TransferToDevice();
	}

	/*
	 *    Reserves the given new size.
	 *    @param	newSize	Size of the new.
	 */

	void reserve(uint newSize) {
		buffer->reserve(newSize);
	}

	/*
	 *    Resizes the given new size.
	 *    @param	newSize	Size of the new.
	 */

	void resize(uint newSize) {
		buffer->resize(newSize);
	}

	void push_back(const T & v) {
		buffer->push_back(v);
	}

	int size() const {
		return buffer->size();
	}

	/*
	 *    Gets the data.
	 *    @return	Null if it fails, else the data.
	 */

	T* data() {
		return buffer->data();
	}

	/*
	 *    Gets the begin.
	 *    @return	Null if it fails, else a pointer to a T.
	 */

	T* begin() {
		return buffer->begin();
	}

	/*
	 *    Gets the begin.
	 *    @return	An Iterator.
	 */

	T* front() {
		return buffer->front();
	}

	/*
	*    Gets the end.
	*    @return	An Iterator.
	*/

	T* end() {
		return buffer->end();
	}

	/*
	 *    Gets the end.
	 *    @return	An Iterator.
	 */

	T* back() {
		return buffer->back();
	}

	/*
	 *    T* casting operator.
	 *    @return	The device data. Facilitates passing in a GPUBuffer object to a kernal
	 */

	operator T*() const {
		return *buffer;
	}

	/*
	 *    Array indexer operator.
	 *    @param	i	Zero-based index of the.
	 *    @return	The indexed value.
	 */

	T operator [](int i) const {
		return (*buffer)[i];
	}

	/*
	 *    Array indexer operator.
	 *    @param	i	Zero-based index of the.
	 *    @return	The indexed value.
	 */

	T & operator [](int i) {
		return (*buffer)[i];
	}

	GPUBuffer<T>& operator=(const GPUBuffer<T>& rhs) {

		this->buffer = rhs.buffer;
		return *this;

	}

private:

	GPUBufferBase<T>* buffer;

};