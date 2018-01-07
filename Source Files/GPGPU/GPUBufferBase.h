//Pure virtual function for the basis of all GPUBuffers

#pragma once

#include "GPGPU/GPUDevice.h"
#include "glm/glm.hpp"
#include <cstring>

/*
*    Buffer for gpu/cpu communication and storage.
*    @tparam	T	Generic type parameter.
*/


template <class T>
class GPUBufferBase {

public:

	/*
	*    Constructor.
	*    @param [in,out]	deviceIn		The device in.
	*    @param 		 	_objectCount	(Optional) Number of objects.
	*/

	GPUBufferBase(const GPUDevice& deviceIn, uint _size = 0){

		host_capacity = _size;
		hostData = new T[host_capacity];
		for (auto i = 0; i < _size; ++i)
			hostData[i] = T();
		host_size = _size;
		flags = 0;

	}

	GPUBufferBase(const GPUDevice& deviceIn, GPUBufferBase<T>& other) {

		host_capacity = other.host_capacity;
		hostData = new T[host_capacity];
		for (auto i = 0; i < other.host_size; ++i)
			hostData[i] = other.hostData[i];
		host_size = other.host_size;

		flags = other.flags;
	}

	/* Destructor. */
	virtual ~GPUBufferBase() {
		delete[] hostData;
	}

	/*
	*    Transfer to host.
	*    @param [in,out]	device	The device.
	*/

	virtual void TransferToHost() = 0;

	/*
	*    Transfer to device.
	*    @param [in,out]	device	The device.
	*/

	virtual void TransferToDevice() = 0;

	/*
	*    Reserves the given new size.
	*    @param	newCapacity	Size of the new.
	*/

	virtual void reserve(uint newCapacity) {

		if (newCapacity > host_capacity) {
			host_capacity = newCapacity;
			reallocate();
		}

	}

	/*
	*    Resizes the given new size.
	*    @param	newSize	Size of the new.
	*/

	virtual void resize(uint newSize) {

		if (newSize > host_size) {
			if (newSize > host_capacity) {
				host_capacity = newSize;
				reallocate();
			}
		}
		else {
			for (auto i = newSize; i < host_size; ++i)
				hostData[i].~T();
		}
		host_size = newSize;

	}

	void push_back(const T& v) {

		if (host_size == host_capacity) {
			host_capacity = glm::max(host_capacity * 2, 1u);
			reallocate();
		}
		hostData[host_size] = v;
		++host_size;
	}

	int HostSize() const {
		return host_size;
	}

	int DeviceSize() const {
		return device_size;
	}

	int HostCapacity() const {
		return host_capacity;
	}

	int DeviceCapacity() const {
		return device_capacity;
	}

	/*
	 *    Gets the data.
	 *    @return	Null if it fails, else the data.
	 */

	T* data() {
		return hostData;
	}

	T * device_data() noexcept {
		return deviceData;
	}

	const T * device_data() const noexcept {
		return deviceData;
	}

	/*
	 *    Gets the begin.
	 *    @return	An T*.
	 */

	T* front() {
		return hostData;
	}

	/*
	*    Gets the begin.
	*    @return	Null if it fails, else a pointer to a T.
	*/

	T* begin() {
		return &hostData[0];
	}

	/*
	*    Gets the end.
	*    @return	An T*.
	*/

	T* end() {
		return &hostData[host_size];
	}

	/*
	 *    Gets the end.
	 *    @return	An T*.
	 */

	T* back() {
		return hostData + host_size - 1;
	}

protected:

	T* hostData;	// Information describing the host
	T* deviceData;  // Information describing the device

	uint host_size = 0;   // Number of objects
	uint host_capacity = 1;	// The capacity
	uint device_size = 0;   // Number of objects
	uint device_capacity = 0;	// The capacity
	uint flags; // The flags

				//operator overloads

private: 

	void reallocate() {
		T* temp = new T[host_capacity];
		std::memcpy(temp, hostData, host_size * sizeof(T));
		delete[] hostData;
		hostData = temp;
	}

public:

	typedef GPUBufferBase<T> * iterator;
	typedef const GPUBufferBase<T> * const_iterator;


	/*
	*    Array indexer operator.
	*    @param	i	Zero-based index of the.
	*    @return	The indexed value.
	*/

	T operator [](int i) const {
		return hostData[i];
	}

	/*
	*    Array indexer operator.
	*    @param	i	Zero-based index of the.
	*    @return	The indexed value.
	*/

	T & operator [](int i) {
		return hostData[i];
	}

	virtual GPUBufferBase& operator=(GPUBufferBase& rhs) {

		swap(rhs);
		return *this;

	}

	void swap(GPUBufferBase& s) throw() // Also see non-throwing swap idiom
	{

		this->host_size = host_size;
		this->host_capacity = host_capacity;

		this->device_size = device_size;
		this->device_capacity = device_capacity;

		this->flags = flags;

		this->hostData = hostData;
		this->deviceData = deviceData;

	}

};