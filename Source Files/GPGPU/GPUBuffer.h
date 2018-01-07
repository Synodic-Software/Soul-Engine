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

		if (deviceIn.GetAPI() == CUDA) {
			buffer = new CUDABuffer<T>(deviceIn, _size);
		}
		else if (deviceIn.GetAPI() == OpenCL) {
			buffer = new OpenCLBuffer<T>(deviceIn, _size);
		}

	}

	GPUBuffer(uint _size = 0) {

		buffer = nullptr;

	}

	/* Destructor. */
	~GPUBuffer() {
		delete buffer;
	}

	void TransferDevice(const GPUDevice& deviceIn) {

		GPUBufferBase<T>* temp;

		if (deviceIn.GetAPI() == CUDA) {
			if (buffer) {
				temp = new CUDABuffer<T>(deviceIn, *buffer);
			}
			else {
				temp = new CUDABuffer<T>(deviceIn, 0);
			}
		}
		else if (deviceIn.GetAPI() == OpenCL) {
			if (buffer) {
				temp = new OpenCLBuffer<T>(deviceIn, *buffer);
			}
			else {
				temp = new CUDABuffer<T>(deviceIn, 0);
			}
		}

		delete buffer;
		buffer = temp;
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

	T * device_data() noexcept {
		return buffer->device_data();
	}

	const T * device_data() const noexcept {
		return buffer->device_data();
	}

	typedef GPUBufferBase<T> * iterator;
	typedef const GPUBufferBase<T> * const_iterator;

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

	int HostSize() const {
		return buffer->HostSize();
	}

	int HostCapacity() const {
		return buffer->HostCapacity();
	}

	int DeviceSize() const {
		return buffer->DeviceSize();
	}

	int DeviceCapacity() const {
		return buffer->DeviceCapacity();
	}

	/*
	*    Gets the data.
	*    @return	Null if it fails, else the data.
	*/

	T* data() {
		return buffer->data();
	}

private:

	GPUBufferBase<T>* buffer;
	
};