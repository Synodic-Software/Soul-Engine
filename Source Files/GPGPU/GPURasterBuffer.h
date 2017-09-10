#pragma once

#include "GPUBuffer.h"
#include "GPGPU\GPUDevice.h"

#include "GPGPU\CUDA\CUDARasterBuffer.h"
#include "GPGPU\OpenCL\OpenCLRasterBuffer.h"

/* Buffer for GPU raster. */
template <class T>
class GPURasterBuffer{

public:
	/* Default constructor. */
	GPURasterBuffer(GPUDevice& _device, uint _objectCount = 0) {

		if (_device.api == CUDA) {
			buffer = new CUDARasterBuffer<T>(_device, _objectCount);
		}
		else if (_device.api == OpenCL) {
			buffer = new OpenCLRasterBuffer<T>(_device, _objectCount);
		}

	}
	/* Destructor. */
	~GPURasterBuffer() {
		delete buffer;
	}

	/* Map resources. */
	void MapResources() {
		buffer->MapResources();
	}

	/* Unmap resources. */
	void UnmapResources() {
		buffer->UnmapResources();
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
	 *    Bind data.
	 *    @param	parameter1	The first parameter.
	 */

	void BindData(uint pos) {
		buffer->BindData(pos);
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

protected:

private:

	GPURasterBufferBase<T>* buffer;

};