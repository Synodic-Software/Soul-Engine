//Pure virtual function for the basis of all GPUBuffers

#pragma once

#include "GPGPU/GPUDevice.h"
#include "glm/glm.hpp"

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

	GPUBufferBase(const GPUDevice& deviceIn, uint _size = 0) {

		host_size = _size;
		host_capacity = _size;

		device_size = _size;
		device_capacity = _size;

		flags = 0;

		hostData = nullptr;
		deviceData = nullptr;

	}

	GPUBufferBase(const GPUDevice& deviceIn, GPUBufferBase<T>& other) {

		host_size = other.size();
		host_capacity = other.capacity();

		flags = other.flags;

		void* raw = operator new[](host_capacity * sizeof(T));
		hostData = static_cast<T*>(raw);
		std::copy(other.data(), other.data() + other.size(), hostData);

		//the specific derived class transfers the device array

	}

	/* Destructor. */
	virtual ~GPUBufferBase() {
		delete hostData;
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

			//allocate the new size
			void* raw = new unsigned char[newCapacity * sizeof(T)];
			T* data = static_cast<T*>(raw);

			uint lSize = newCapacity < host_size ? newCapacity : host_size;

			//move all previous data
			for (uint i = 0; i < lSize; i++) {
				data[i] = std::move(hostData[i]);
			}

			host_capacity = newCapacity;

			//deallocate formerly used memory
			if (hostData) {
				delete[] hostData;
			}

			hostData = data; //reassign the data pointer
		}

	}

	/*
	*    Resizes the given new size.
	*    @param	newSize	Size of the new.
	*/

	virtual void resize(uint newSize) {

		GPUBufferBase::reserve(newSize);
		host_size = newSize;  //change host size.

	}

	void push_back(const T & v) {

		if (host_size >= host_capacity) {
			reserve(glm::max(host_capacity * 2, 1u));
		}

		hostData[host_size++] = v;

	}

	int size() const {
		return host_size;
	}

	int capacity() const {
		return host_capacity;
	}

	/*
	 *    Gets the data.
	 *    @return	Null if it fails, else the data.
	 */

	T* data() {
		return hostData;
	}

	T* device_data() {
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
		return hostData;
	}

	/*
	*    Gets the end.
	*    @return	An T*.
	*/

	T* end() {
		return hostData + host_size - 1;
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

	uint host_size;   // Number of objects
	uint host_capacity;	// The capacity
	uint device_size;   // Number of objects
	uint device_capacity;	// The capacity
	uint flags; // The flags

				//operator overloads
public:

	/*
	*    T* casting operator.
	*    @return	The device data. Facilitates passing in a GPUBuffer object to a kernal
	*/

	operator T*() const {
		return deviceData;
	}

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