#pragma once

/*
*    The wrapper class which passes all functions to the actual buffer,
*    a user friendly factory class encapsulating the polymorphic buffer
*/


#include "GPGPU/GPUDevice.h"
#include "GPGPU/DeviceBuffer.h"

#include "GPGPU/CUDA/CUDABuffer.h"
#include "GPGPU/OpenCL/OpenCLBuffer.h"

#include "Utility/Logger.h"

/*
 *    Buffer for device/host(GPU/CPU) communication and storage.
 *    @tparam	T	Generic type parameter.
 */

 //TODO add initializer list functions

template <class T>
class ComputeBuffer {

public:

	//Types

	typedef std::unique_ptr<DeviceBuffer<T>>      device_pointer;
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

	ComputeBuffer();
	explicit ComputeBuffer(const GPUDevice&, size_type);
	ComputeBuffer(const GPUDevice&, size_type, const T&);
	ComputeBuffer(const ComputeBuffer&);
	ComputeBuffer(ComputeBuffer&&) noexcept;

	~ComputeBuffer() = default;

	ComputeBuffer<T>& operator= (ComputeBuffer<T>&&) noexcept;
	ComputeBuffer<T>& operator= (const ComputeBuffer<T>&);

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

	void Move(const GPUDevice&);

	void TransferToHost();
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

	T* DeviceData();
	const T* DeviceData() const;


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

	void Swap(ComputeBuffer<T>&);

	void Clear() noexcept;


private:

	std::vector<T> hostBuffer;
	device_pointer deviceBuffer;

	//TODO implement bit-per-T member bitset

	//TODO create usage flags
	uint8 flags;
};

template <class T>
ComputeBuffer<T>::ComputeBuffer() :
	flags(0)
{
}

template <class T>
ComputeBuffer<T>::ComputeBuffer(const GPUDevice& device, size_type n) :
	hostBuffer(n),
	flags(0)
{

	if (device.GetAPI() == CUDA) {
		deviceBuffer.reset(new CUDABuffer<T>(device, n));
	}
	else if (device.GetAPI() == OpenCL) {
		deviceBuffer.reset(new OpenCLBuffer<T>(device, n));
	}

}
template <class T>
ComputeBuffer<T>::ComputeBuffer(const GPUDevice& device, size_type n, const T &val) :
	hostBuffer(n, val),
	flags(0)
{
	if (device.GetAPI() == CUDA) {
		deviceBuffer.reset(new CUDABuffer<T>(device, n, val));
	}
	else if (device.GetAPI() == OpenCL) {
		deviceBuffer.reset(new OpenCLBuffer<T>(device, n, val));
	}
}


template <class T>
ComputeBuffer<T>::ComputeBuffer(const ComputeBuffer& other)
{
	*this = other;
}

template <class T>
ComputeBuffer<T>::ComputeBuffer(ComputeBuffer<T>&& other) noexcept
{
	*this = std::move(other);
}

template <class T>
ComputeBuffer<T>& ComputeBuffer<T>::operator= (ComputeBuffer<T>&& other) noexcept
{
	hostBuffer = std::move(other.hostBuffer);
	deviceBuffer = std::move(other.deviceBuffer);
	flags = std::move(other.flags);

	return *this;
}

template <class T>
ComputeBuffer<T>& ComputeBuffer<T>::operator= (const ComputeBuffer<T>& other)
{
	hostBuffer = other.hostBuffer;
	deviceBuffer = other.deviceBuffer;
	flags = other.flags;
	return *this;
}

template <class T>
typename ComputeBuffer<T>::iterator ComputeBuffer<T>::begin() noexcept
{
	return hostBuffer.begin();
}

template <class T>
typename ComputeBuffer<T>::const_iterator ComputeBuffer<T>::cbegin() const noexcept
{
	return hostBuffer.cbegin();
}

template <class T>
typename ComputeBuffer<T>::iterator ComputeBuffer<T>::end() noexcept
{
	return hostBuffer.end();
}

template <class T>
typename ComputeBuffer<T>::const_iterator ComputeBuffer<T>::cend() const noexcept
{
	return hostBuffer.cend();
}
template <class T>
typename ComputeBuffer<T>::reverse_iterator ComputeBuffer<T>::rbegin() noexcept
{
	return hostBuffer.rbegin();
}
template <class T>
typename ComputeBuffer<T>::const_reverse_iterator ComputeBuffer<T>::crbegin() const noexcept
{
	return hostBuffer.crbegin();
}
template <class T>
typename ComputeBuffer<T>::reverse_iterator ComputeBuffer<T>::rend() noexcept
{
	return hostBuffer.rend();
}
template <class T>
typename ComputeBuffer<T>::const_reverse_iterator ComputeBuffer<T>::crend() const noexcept
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
void ComputeBuffer<T>::Move(const GPUDevice& device) {

	if (deviceBuffer) {
		deviceBuffer->Move(device);
	}
	else
	{
		if (device.GetAPI() == CUDA) {
			deviceBuffer.reset(device_pointer(new CUDABuffer<T>(device, hostBuffer)));
		}
		else if (device.GetAPI() == OpenCL)
		{
			deviceBuffer.reset(device_pointer(new OpenCLBuffer<T>(device, hostBuffer)));
		}
	}

}

template <class T>
void ComputeBuffer<T>::TransferToHost()
{
	deviceBuffer->TransferToHost(hostBuffer);
}

template <class T>
void ComputeBuffer<T>::TransferToDevice()
{
	deviceBuffer->TransferToDevice(hostBuffer);
}

template <class T>
typename ComputeBuffer<T>::reference ComputeBuffer<T>::At(size_type pos)
{
	return hostBuffer.at(pos);
}

template <class T>
typename ComputeBuffer<T>::const_reference ComputeBuffer<T>::At(size_type pos) const
{
	return hostBuffer.at(pos);
}

template <class T>
typename ComputeBuffer<T>::reference ComputeBuffer<T>::operator[](size_type pos)
{
	return hostBuffer[pos];
}

template <class T>
typename ComputeBuffer<T>::const_reference ComputeBuffer<T>::operator[](size_type pos) const
{
	return hostBuffer[pos];
}

template <class T>
typename ComputeBuffer<T>::reference ComputeBuffer<T>::Front()
{
	return hostBuffer.front();
}

template <class T>
typename ComputeBuffer<T>::const_reference ComputeBuffer<T>::Front() const
{
	return hostBuffer.front();
}

template <class T>
typename ComputeBuffer<T>::reference ComputeBuffer<T>::Back()
{
	return hostBuffer.back();
}

template <class T>
typename ComputeBuffer<T>::const_reference ComputeBuffer<T>::Back() const
{
	return hostBuffer.back();
}

template <class T>
T* ComputeBuffer<T>::HostData() noexcept {
	return hostBuffer.data();
}

template <class T>
const T* ComputeBuffer<T>::HostData() const noexcept {
	return hostBuffer.data();
}

template <class T>
T* ComputeBuffer<T>::DeviceData() {
	return deviceBuffer->Data();
}

template <class T>
const T* ComputeBuffer<T>::DeviceData() const {
	return deviceBuffer->Data();
}

template <class T>
bool ComputeBuffer<T>::Empty() const noexcept
{
	return EmptyHost() && EmptyDevice();
}

template <class T>
bool ComputeBuffer<T>::EmptyHost() const noexcept
{
	return hostBuffer.empty();
}

template <class T>
bool ComputeBuffer<T>::EmptyDevice() const noexcept
{
	return deviceBuffer->Empty();
}

template <class T>
typename ComputeBuffer<T>::size_type ComputeBuffer<T>::SizeHost() const noexcept
{
	return hostBuffer.size();
}

template <class T>
typename ComputeBuffer<T>::size_type ComputeBuffer<T>::SizeDevice() const noexcept
{
	return deviceBuffer->Size();
}

template <class T>
typename ComputeBuffer<T>::size_type ComputeBuffer<T>::MaxSize() const noexcept
{
	return glm::min(hostBuffer.max_size(), deviceBuffer->MaxSize());
}

template <class T>
typename ComputeBuffer<T>::size_type ComputeBuffer<T>::HostCapacity() const noexcept
{
	return hostBuffer.capacity();
}

template <class T>
typename ComputeBuffer<T>::size_type ComputeBuffer<T>::DeviceCapacity() const noexcept
{
	return deviceBuffer->Capacity();
}

template <class T>
void ComputeBuffer<T>::Resize(size_type n)
{
	ResizeHost(n);
	ResizeDevice(n);
}

template <class T>
void ComputeBuffer<T>::Resize(size_type n, const T& val)
{
	ResizeHost(n, val);
	ResizeDevice(n, val);
}

template <class T>
void ComputeBuffer<T>::ResizeHost(size_type n)
{
	hostBuffer.resize(n);
}

template <class T>
void ComputeBuffer<T>::ResizeHost(size_type n, const T& val)
{
	hostBuffer.resize(n, val);
}

template <class T>
void ComputeBuffer<T>::ResizeDevice(size_type n)
{
	deviceBuffer->Resize(n);
}

template <class T>
void ComputeBuffer<T>::ResizeDevice(size_type n, const T& val)
{
	deviceBuffer->Resize(n, val);
}

template <class T>
void ComputeBuffer<T>::Reserve(size_type n)
{
	ReserveHost(n);
	ReserveDevice(n);
}

template <class T>
void ComputeBuffer<T>::ReserveHost(size_type n)
{
	hostBuffer.reserve(n);
}

template <class T>
void ComputeBuffer<T>::ReserveDevice(size_type n)
{
	deviceBuffer->Reserve(n);
}

template <class T>
void ComputeBuffer<T>::Fit()
{
	FitHost();
	FitDevice();
}

template <class T>
void ComputeBuffer<T>::FitHost()
{
	hostBuffer.shrink_to_fit();
}

template <class T>
void ComputeBuffer<T>::FitDevice()
{
	deviceBuffer->Fit();
}

template <typename T>
template <class ... Args>
void ComputeBuffer<T>::EmplaceBack(Args&& ... args)
{
	hostBuffer.emplace_back(args...);
}

template <class T>
void ComputeBuffer<T>::PushBack(const T& val)
{
	hostBuffer.push_back(val);
}

template <class T>
void ComputeBuffer<T>::PushBack(T&& val)
{
	hostBuffer.push_back(val);
}

template <class T>
void ComputeBuffer<T>::PopBack()
{
	hostBuffer.pop_back();
}

template <typename T>
template <class ... Args>
typename ComputeBuffer<T>::iterator ComputeBuffer<T>::Emplace(const_iterator itr, Args&& ... args)
{
	return hostBuffer.emplace(itr, args...);
}

template <class T>
typename ComputeBuffer<T>::iterator ComputeBuffer<T>::Insert(const_iterator itr, const T& val)
{
	return hostBuffer.insert(itr, val);
}

template <class T>
typename ComputeBuffer<T>::iterator ComputeBuffer<T>::Insert(const_iterator itr, T&& val)
{
	return hostBuffer.insert(itr, val);
}

template <class T>
typename ComputeBuffer<T>::iterator ComputeBuffer<T>::Insert(const_iterator itr, size_type n, const T& val)
{
	return hostBuffer.insert(itr, n, val);
}

template <typename T>
template <class InputIt>
typename ComputeBuffer<T>::iterator ComputeBuffer<T>::Insert(const_iterator itr, InputIt begin, InputIt end)
{
	return hostBuffer.insert(itr, begin, end);
}

template <class T>
typename ComputeBuffer<T>::iterator ComputeBuffer<T>::Erase(const_iterator itr)
{
	return hostBuffer.erase(itr);
}

template <class T>
typename ComputeBuffer<T>::iterator ComputeBuffer<T>::Erase(const_iterator begin, const_iterator end)
{
	return hostBuffer.erase(begin, end);
}

template <class T>
void ComputeBuffer<T>::Swap(ComputeBuffer<T>& other)
{
	hostBuffer.swap(other.hostBuffer);
	deviceBuffer.swap(other.deviceBuffer);
	std::swap(flags, other.flags);
}

template <class T>
void ComputeBuffer<T>::Clear() noexcept
{
	hostBuffer.clear();
}