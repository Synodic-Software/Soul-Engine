#pragma once

/*
*    The wrapper class which passes all functions to the actual buffer,
*    a user friendly factory class encapsulating the polymorphic buffer
*/


#include "GPGPU/GPUDevice.h"
#include "GPGPU/GPUBufferBase.h"

#include "GPGPU/CUDA/CUDABuffer.h"
#include "GPGPU/OpenCL/OpenCLBuffer.h"

#include "Utility/Logger.h"

/*
 *    Buffer for device/host(GPU/CPU) communication and storage.
 *    @tparam	T	Generic type parameter.
 */

 //TODO add initializer list functions

template <class T>
class GPUBuffer {

public:

	//Types

	typedef std::unique_ptr<GPUBufferBase<T>>     device_pointer;
	typedef T                                     value_type;
	typedef T&                                    reference;
	typedef const T&                              const_reference;
	typedef T*                                    pointer;
	typedef const T*                              const_pointer;
	typedef T*                                    iterator;
	typedef const T*                              const_iterator;
	typedef std::reverse_iterator<iterator>       reverse_iterator;
	typedef std::reverse_iterator<const_iterator> const_reverse_iterator;
	typedef ptrdiff_t                             difference_type;
	typedef uint		                          size_type;

	//Construction and Destruction 

	GPUBuffer();
	explicit GPUBuffer(const std::vector<GPUDevice>&, size_type);
	GPUBuffer(const std::vector<GPUDevice>&, size_type, const T&);
	GPUBuffer(const GPUBuffer&);
	GPUBuffer(GPUBuffer&&) noexcept;

	~GPUBuffer() = default;

	GPUBuffer<T>& operator= (GPUBuffer<T>&&) noexcept;
	GPUBuffer<T>& operator= (const GPUBuffer<T>&);

	//Iterators

	iterator begin() noexcept;
	const_iterator cbegin() const noexcept;
	iterator end() noexcept;
	const_iterator cend() const noexcept;
	reverse_iterator rbegin() noexcept;
	const_reverse_iterator crbegin() const noexcept;
	reverse_iterator rend() noexcept;
	const_reverse_iterator crend() const noexcept;


	//Data Migration

	void Move(const std::vector<GPUDevice>&);

	void TransferToHost(size_type pos);
	void TransferToDevice();


	//Element Access

	reference At(size_type);
	const_reference At(size_type) const;

	reference operator[](size_type);
	const_reference operator[](size_type) const;

	reference Front();
	const_reference Front() const;

	reference Back();
	const_reference Back() const;

	T* HostData() noexcept;
	const T* HostData() const noexcept;

	T* DeviceData() noexcept;
	const T* DeviceData() const noexcept;


	//Capacity

	bool Empty() const noexcept;
	bool EmptyHost() const noexcept;
	bool EmptyDevice() const noexcept;

	size_type SizeHost() const noexcept;
	size_type SizeDevice() const noexcept;

	size_type MaxSize() const noexcept;

	size_type HostCapacity() const noexcept;
	size_type DeviceCapacity() const noexcept;

	void Resize(size_type);
	void Resize(size_type, const T&);
	void ResizeHost(size_type);
	void ResizeHost(size_type, const T&);
	void ResizeDevice(size_type);
	void ResizeDevice(size_type, const T&);

	void Reserve(size_type);
	void ReserveHost(size_type);
	void ReserveDevice(size_type);

	void Fit();
	void FitHost();
	void FitDevice();

	//Modifiers
	template <class ... Args> void EmplaceBack(Args&& ...);
	void PushBack(const T&);
	void PushBack(T&&);
	void PopBack();

	template <class ... Args>
	iterator Emplace(const_iterator, Args&& ...);

	iterator Insert(const_iterator, const T&);
	iterator Insert(const_iterator, T&&);
	iterator Insert(const_iterator, size_type, const T&);

	template <class InputIt>
	iterator Insert(const_iterator, InputIt, InputIt);

	iterator Erase(const_iterator);
	iterator Erase(const_iterator, const_iterator);

	void Swap(GPUBuffer<T>&);

	void Clear() noexcept;


private:

	std::vector<T> hostBuffer;
	std::vector<device_pointer> deviceBuffers;

	//TODO implement bit-per-T member bitset

	//TODO create usage flags
	uint8 flags;
};

template <class T>
GPUBuffer<T>::GPUBuffer() :
	flags(0)
{
}

template <class T>
GPUBuffer<T>::GPUBuffer(const std::vector<GPUDevice>& devices, size_type n) :
	hostBuffer(n),
	flags(0)
{

	for (const auto& device : devices) {
		if (device.GetAPI() == CUDA) {
			deviceBuffers.push_back(device_pointer(new CUDABuffer<T>(device, n)));
		}
		else if (device.GetAPI() == OpenCL) {
			deviceBuffers.push_back(device_pointer(new OpenCLBuffer<T>(device, n)));
		}
	}

}
template <class T>
GPUBuffer<T>::GPUBuffer(const std::vector<GPUDevice>& devices, size_type n, const T &val) :
	hostBuffer(n, val),
	flags(0)
{
	for (const auto& device : devices) {
		if (device.GetAPI() == CUDA) {
			deviceBuffers.push_back(device_pointer(new CUDABuffer<T>(device, n, val)));
		}
		else if (device.GetAPI() == OpenCL) {
			deviceBuffers.push_back(device_pointer(new OpenCLBuffer<T>(device, n, val)));
		}
	}
}


template <class T>
GPUBuffer<T>::GPUBuffer(const GPUBuffer& other)
{
	*this = other;
}

template <class T>
GPUBuffer<T>::GPUBuffer(GPUBuffer<T>&& other) noexcept
{
	*this = std::move(other);
}

template <class T>
GPUBuffer<T>& GPUBuffer<T>::operator= (GPUBuffer<T>&& other) noexcept
{
	hostBuffer = std::move(other.hostBuffer);
	deviceBuffers = std::move(other.deviceBuffers);
	flags = std::move(other.flags);

	return *this;
}

template <class T>
GPUBuffer<T>& GPUBuffer<T>::operator= (const GPUBuffer<T>& other)
{
	hostBuffer = other.hostBuffer;
	throw std::exception("Not implemented");
	//TODO implement copy of device buffers
	//deviceBuffers = copy;
	flags = other.flags;
	return *this;
}

template <class T>
typename GPUBuffer<T>::iterator GPUBuffer<T>::begin() noexcept
{
	return hostBuffer.begin();
}

template <class T>
typename GPUBuffer<T>::const_iterator GPUBuffer<T>::cbegin() const noexcept
{
	return hostBuffer.cbegin();
}

template <class T>
typename GPUBuffer<T>::iterator GPUBuffer<T>::end() noexcept
{
	return hostBuffer.end();
}

template <class T>
typename GPUBuffer<T>::const_iterator GPUBuffer<T>::cend() const noexcept
{
	return hostBuffer.cend();
}
template <class T>
typename GPUBuffer<T>::reverse_iterator GPUBuffer<T>::rbegin() noexcept
{
	return hostBuffer.rbegin();
}
template <class T>
typename GPUBuffer<T>::const_reverse_iterator GPUBuffer<T>::crbegin() const noexcept
{
	return hostBuffer.crbegin();
}
template <class T>
typename GPUBuffer<T>::reverse_iterator GPUBuffer<T>::rend() noexcept
{
	return hostBuffer.rend();
}
template <class T>
typename GPUBuffer<T>::const_reverse_iterator GPUBuffer<T>::crend() const noexcept
{
	return hostBuffer.crend();
}

/*
 * Move the buffer to the specified device
 *
 * @tparam	T	Generic type parameter.
 * @param	devices	The devices to transfer to.
 */
template <class T>
void GPUBuffer<T>::Move(const std::vector<GPUDevice>& devices) {

	std::unique_ptr<GPUBufferBase<T>> temp;

	//for each DeviceBuffer that does not exist, create it

	for (const auto& device : devices) {
		for (const auto& devicePointer : deviceBuffers) {
			if () {
				if (device.GetAPI() == CUDA) {
					deviceBuffers.push_back(device_pointer(new CUDABuffer<T>(device, hostBuffer)));
				}
				else if (device.GetAPI() == OpenCL)
				{
					deviceBuffers.push_back(device_pointer(new OpenCLBuffer<T>(device, hostBuffer)));
				}
			}
		}
	}

}

template <class T>
void GPUBuffer<T>::TransferToHost(size_type pos)
{
	deviceBuffers[pos]->TransferToHost(hostBuffer);
}

template <class T>
void GPUBuffer<T>::TransferToDevice()
{
	for (const auto& devicePointer : deviceBuffers) {
		devicePointer->TransferToDevice(hostBuffer);
	}
}

template <class T>
typename GPUBuffer<T>::reference GPUBuffer<T>::At(size_type pos)
{
	return hostBuffer.at(pos);
}

template <class T>
typename GPUBuffer<T>::const_reference GPUBuffer<T>::At(size_type pos) const
{
	return hostBuffer.at(pos);
}

template <class T>
typename GPUBuffer<T>::reference GPUBuffer<T>::operator[](size_type pos)
{
	return hostBuffer[pos];
}

template <class T>
typename GPUBuffer<T>::const_reference GPUBuffer<T>::operator[](size_type pos) const
{
	return hostBuffer[pos];
}

template <class T>
typename GPUBuffer<T>::reference GPUBuffer<T>::Front()
{
	return hostBuffer.front();
}

template <class T>
typename GPUBuffer<T>::const_reference GPUBuffer<T>::Front() const
{
	return hostBuffer.front();
}

template <class T>
typename GPUBuffer<T>::reference GPUBuffer<T>::Back()
{
	return hostBuffer.back();
}

template <class T>
typename GPUBuffer<T>::const_reference GPUBuffer<T>::Back() const
{
	return hostBuffer.back();
}

template <class T>
T* GPUBuffer<T>::HostData() noexcept {
	return hostBuffer.data();
}

template <class T>
const T* GPUBuffer<T>::HostData() const noexcept {
	return hostBuffer.data();
}

template <class T>
T* GPUBuffer<T>::DeviceData() noexcept {
	//TODO implement
	throw std::exception("Not implemented");
}

template <class T>
const T* GPUBuffer<T>::DeviceData() const noexcept {
	//TODO implement
	throw std::exception("Not implemented");
}

template <class T>
bool GPUBuffer<T>::Empty() const noexcept
{
	return EmptyHost() && EmptyDevice();
}

template <class T>
bool GPUBuffer<T>::EmptyHost() const noexcept
{
	return hostBuffer.empty();
}

template <class T>
bool GPUBuffer<T>::EmptyDevice() const noexcept
{
	//TODO implement
	throw std::exception("Not implemented");
}

template <class T>
typename GPUBuffer<T>::size_type GPUBuffer<T>::SizeHost() const noexcept
{
	return hostBuffer.size();
}

template <class T>
typename GPUBuffer<T>::size_type GPUBuffer<T>::SizeDevice() const noexcept
{
	//TODO implement
	throw std::exception("Not implemented");
}

template <class T>
typename GPUBuffer<T>::size_type GPUBuffer<T>::MaxSize() const noexcept
{
	//TODO implement
	throw std::exception("Not implemented");
}

template <class T>
typename GPUBuffer<T>::size_type GPUBuffer<T>::HostCapacity() const noexcept
{
	return hostBuffer.capacity();
}

template <class T>
typename GPUBuffer<T>::size_type GPUBuffer<T>::DeviceCapacity() const noexcept
{
	//TODO implement
	throw std::exception("Not implemented");
}

template <class T>
void GPUBuffer<T>::Resize(size_type n)
{
	ResizeHost(n);
	ResizeDevice(n);
}

template <class T>
void GPUBuffer<T>::Resize(size_type n, const T& val)
{
	ResizeHost(n, val);
	ResizeDevice(n, val);
}

template <class T>
void GPUBuffer<T>::ResizeHost(size_type n)
{
	hostBuffer.resize(n);
}

template <class T>
void GPUBuffer<T>::ResizeHost(size_type n, const T& val)
{
	hostBuffer.resize(n, val);
}

template <class T>
void GPUBuffer<T>::ResizeDevice(size_type n)
{
	//TODO implement
	throw std::exception("Not implemented");
}

template <class T>
void GPUBuffer<T>::ResizeDevice(size_type n, const T& val)
{
	//TODO implement
	throw std::exception("Not implemented");
}

template <class T>
void GPUBuffer<T>::Reserve(size_type n)
{
	ReserveHost(n);
	ReserveDevice(n);
}

template <class T>
void GPUBuffer<T>::ReserveHost(size_type n)
{
	hostBuffer.reserve(n);
}

template <class T>
void GPUBuffer<T>::ReserveDevice(size_type n)
{
	//TODO implement
	throw std::exception("Not implemented");
}

template <class T>
void GPUBuffer<T>::Fit()
{
	FitHost();
	FitDevice();
}

template <class T>
void GPUBuffer<T>::FitHost()
{
	hostBuffer.shrink_to_fit();
}

template <class T>
void GPUBuffer<T>::FitDevice()
{
	//TODO implement
	throw std::exception("Not implemented");
}

template <typename T>
template <class ... Args> 
void GPUBuffer<T>::EmplaceBack(Args&& ... args)
{
	hostBuffer.emplace_back(args);
}

template <class T>
void GPUBuffer<T>::PushBack(const T& val)
{
	hostBuffer.push_back(val);
}

template <class T>
void GPUBuffer<T>::PushBack(T&& val)
{
	hostBuffer.push_back(val);
}

template <class T>
void GPUBuffer<T>::PopBack()
{
	hostBuffer.pop_back();
}

template <typename T>
template <class ... Args>
typename GPUBuffer<T>::iterator GPUBuffer<T>::Emplace(const_iterator itr, Args&& ... args)
{
	return hostBuffer.emplace(itr, args);
}

template <class T>
typename GPUBuffer<T>::iterator GPUBuffer<T>::Insert(const_iterator itr, const T& val)
{
	return hostBuffer.insert(itr, val);
}

template <class T>
typename GPUBuffer<T>::iterator GPUBuffer<T>::Insert(const_iterator itr, T&& val)
{
	return hostBuffer.insert(itr, val);
}

template <class T>
typename GPUBuffer<T>::iterator GPUBuffer<T>::Insert(const_iterator itr, size_type n, const T& val)
{
	return hostBuffer.insert(itr, n, val);
}

template <typename T>
template <class InputIt>
typename GPUBuffer<T>::iterator GPUBuffer<T>::Insert(const_iterator itr, InputIt begin, InputIt end)
{
	return hostBuffer.insert(itr, begin, end);
}

template <class T>
typename GPUBuffer<T>::iterator GPUBuffer<T>::Erase(const_iterator itr)
{
	return hostBuffer.erase(itr);
}

template <class T>
typename GPUBuffer<T>::iterator GPUBuffer<T>::Erase(const_iterator begin, const_iterator end)
{
	return hostBuffer.erase(begin, end);
}

template <class T>
void GPUBuffer<T>::Swap(GPUBuffer<T>& other)
{
	//TODO implement
	throw std::exception("Not implemented");
}

template <class T>
void GPUBuffer<T>::Clear() noexcept
{
	hostBuffer.clear();
}