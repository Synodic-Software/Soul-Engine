#pragma once

/*
*    The wrapper class which passes all functions to the actual buffer,
*    a user friendly factory class encapsulating the polymorphic buffer
*/


#include "Compute/GPUDevice.h"

#include "Utility/Logger.h"

/*
 *    Buffer for device/host(GPU/CPU) communication and storage.
 *    @tparam	T	Generic type parameter.
 */

 //TODO add initializer list functions

template <class T>
class AbstractComputeBuffer {

public:

	//Types

	typedef typename std::vector<T>::value_type             value_type;
	typedef typename std::vector<T>::reference              reference;
	typedef typename std::vector<T>::const_reference        const_reference;
	typedef typename std::vector<T>::pointer                pointer;
	typedef typename std::vector<T>::const_pointer          const_pointer;
	typedef typename std::vector<T>::iterator               iterator;
	typedef typename std::vector<T>::const_iterator         const_iterator;
	typedef typename std::vector<T>::reverse_iterator       reverse_iterator;
	typedef typename std::vector<T>::const_reverse_iterator const_reverse_iterator;
	typedef typename std::vector<T>::difference_type        difference_type;
	typedef uint	                                        size_type;

	//Construction and Destruction 

	AbstractComputeBuffer();
	AbstractComputeBuffer(const GPUDevice&);
	AbstractComputeBuffer(const GPUDevice&, size_type);
	AbstractComputeBuffer(const GPUDevice&, size_type, const T&);
	AbstractComputeBuffer(const AbstractComputeBuffer&);
	AbstractComputeBuffer(AbstractComputeBuffer&&) noexcept;

	virtual ~AbstractComputeBuffer() = default;

	AbstractComputeBuffer<T>& operator= (AbstractComputeBuffer<T>&&) noexcept;
	AbstractComputeBuffer<T>& operator= (const AbstractComputeBuffer<T>&);

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

	virtual void Move(const GPUDevice&) = 0;

	virtual void TransferToHost() = 0;
	virtual void TransferToDevice() = 0;


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

	virtual T* DataDevice() = 0;
	virtual const T* DataDevice() const = 0;


	//Capacity

	bool Empty() const noexcept;
	bool EmptyHost() const noexcept;
	virtual bool EmptyDevice() const noexcept = 0;

	size_type SizeHost() const noexcept;
	virtual size_type SizeDevice() const noexcept = 0;

	size_type MaxSize() const noexcept;
	size_type MaxSizeHost() const noexcept;
	virtual size_type MaxSizeDevice() const noexcept = 0;

	size_type HostCapacity() const noexcept;
	virtual size_type DeviceCapacity() const noexcept = 0;

	void Resize(size_type);
	void Resize(size_type, const T&);
	void ResizeHost(size_type);
	void ResizeHost(size_type, const T&);
	virtual void ResizeDevice(size_type) = 0;
	virtual void ResizeDevice(size_type, const T&) = 0;

	void Reserve(size_type);
	void ReserveHost(size_type);
	virtual void ReserveDevice(size_type) = 0;

	void Fit();
	void FitHost();
	virtual void FitDevice() = 0;

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

	

	void Clear() noexcept;

protected:

	void Swap(AbstractComputeBuffer<T>&);

	std::vector<T> hostBuffer;

	//TODO implement bit-per-T member bitset

	//TODO create usage flags
	uint8 flags;
	
};

template <class T>
AbstractComputeBuffer<T>::AbstractComputeBuffer() :
	flags(0)
{
}

template <class T>
AbstractComputeBuffer<T>::AbstractComputeBuffer(const GPUDevice& device) :
	flags(0)
{
}

template <class T>
AbstractComputeBuffer<T>::AbstractComputeBuffer(const GPUDevice& device, size_type n) :
	hostBuffer(n),
	flags(0)
{
}

template <class T>
AbstractComputeBuffer<T>::AbstractComputeBuffer(const GPUDevice& device, size_type n, const T &val) :
	hostBuffer(n, val),
	flags(0)
{
}


template <class T>
AbstractComputeBuffer<T>::AbstractComputeBuffer(const AbstractComputeBuffer& other)
{
	*this = other;
}

template <class T>
AbstractComputeBuffer<T>::AbstractComputeBuffer(AbstractComputeBuffer<T>&& other) noexcept
{
	*this = std::move(other);
}

template <class T>
AbstractComputeBuffer<T>& AbstractComputeBuffer<T>::operator= (AbstractComputeBuffer<T>&& other) noexcept
{
	hostBuffer = std::move(other.hostBuffer);
	flags = std::move(other.flags);

	return *this;
}

template <class T>
AbstractComputeBuffer<T>& AbstractComputeBuffer<T>::operator= (const AbstractComputeBuffer<T>& other)
{

	hostBuffer = other.hostBuffer;
	flags = other.flags;

	return *this;
}

template <class T>
typename AbstractComputeBuffer<T>::iterator AbstractComputeBuffer<T>::begin() noexcept
{
	return hostBuffer.begin();
}

template <class T>
typename AbstractComputeBuffer<T>::const_iterator AbstractComputeBuffer<T>::cbegin() const noexcept
{
	return hostBuffer.cbegin();
}

template <class T>
typename AbstractComputeBuffer<T>::iterator AbstractComputeBuffer<T>::end() noexcept
{
	return hostBuffer.end();
}

template <class T>
typename AbstractComputeBuffer<T>::const_iterator AbstractComputeBuffer<T>::cend() const noexcept
{
	return hostBuffer.cend();
}
template <class T>
typename AbstractComputeBuffer<T>::reverse_iterator AbstractComputeBuffer<T>::rbegin() noexcept
{
	return hostBuffer.rbegin();
}
template <class T>
typename AbstractComputeBuffer<T>::const_reverse_iterator AbstractComputeBuffer<T>::crbegin() const noexcept
{
	return hostBuffer.crbegin();
}
template <class T>
typename AbstractComputeBuffer<T>::reverse_iterator AbstractComputeBuffer<T>::rend() noexcept
{
	return hostBuffer.rend();
}
template <class T>
typename AbstractComputeBuffer<T>::const_reverse_iterator AbstractComputeBuffer<T>::crend() const noexcept
{
	return hostBuffer.crend();
}

template <class T>
typename AbstractComputeBuffer<T>::reference AbstractComputeBuffer<T>::At(size_type pos)
{
	return hostBuffer.at(pos);
}

template <class T>
typename AbstractComputeBuffer<T>::const_reference AbstractComputeBuffer<T>::At(size_type pos) const
{
	return hostBuffer.at(pos);
}

template <class T>
typename AbstractComputeBuffer<T>::reference AbstractComputeBuffer<T>::operator[](size_type pos)
{
	return hostBuffer[pos];
}

template <class T>
typename AbstractComputeBuffer<T>::const_reference AbstractComputeBuffer<T>::operator[](size_type pos) const
{
	return hostBuffer[pos];
}

template <class T>
typename AbstractComputeBuffer<T>::reference AbstractComputeBuffer<T>::Front()
{
	return hostBuffer.front();
}

template <class T>
typename AbstractComputeBuffer<T>::const_reference AbstractComputeBuffer<T>::Front() const
{
	return hostBuffer.front();
}

template <class T>
typename AbstractComputeBuffer<T>::reference AbstractComputeBuffer<T>::Back()
{
	return hostBuffer.back();
}

template <class T>
typename AbstractComputeBuffer<T>::const_reference AbstractComputeBuffer<T>::Back() const
{
	return hostBuffer.back();
}

template <class T>
T* AbstractComputeBuffer<T>::HostData() noexcept {
	return hostBuffer.data();
}

template <class T>
const T* AbstractComputeBuffer<T>::HostData() const noexcept {
	return hostBuffer.data();
}

template <class T>
bool AbstractComputeBuffer<T>::Empty() const noexcept
{
	return EmptyHost() && EmptyDevice();
}

template <class T>
bool AbstractComputeBuffer<T>::EmptyHost() const noexcept
{
	return hostBuffer.empty();
}

template <class T>
typename AbstractComputeBuffer<T>::size_type AbstractComputeBuffer<T>::SizeHost() const noexcept
{
	return static_cast<unsigned int>(hostBuffer.size());
}

template <class T>
typename AbstractComputeBuffer<T>::size_type AbstractComputeBuffer<T>::MaxSize() const noexcept
{
	return glm::min(MaxSizeHost(), MaxSizeDevice());
}

template <class T>
typename AbstractComputeBuffer<T>::size_type AbstractComputeBuffer<T>::MaxSizeHost() const noexcept
{
	return hostBuffer.max_size();
}

template <class T>
typename AbstractComputeBuffer<T>::size_type AbstractComputeBuffer<T>::HostCapacity() const noexcept
{
	return hostBuffer.capacity();
}

template <class T>
void AbstractComputeBuffer<T>::Resize(size_type n)
{
	ResizeHost(n);
	ResizeDevice(n);
}

template <class T>
void AbstractComputeBuffer<T>::Resize(size_type n, const T& val)
{
	ResizeHost(n, val);
	ResizeDevice(n, val);
}

template <class T>
void AbstractComputeBuffer<T>::ResizeHost(size_type n)
{
	hostBuffer.resize(n);
}

template <class T>
void AbstractComputeBuffer<T>::ResizeHost(size_type n, const T& val)
{
	hostBuffer.resize(n, val);
}

template <class T>
void AbstractComputeBuffer<T>::Reserve(size_type n)
{
	ReserveHost(n);
	ReserveDevice(n);
}

template <class T>
void AbstractComputeBuffer<T>::ReserveHost(size_type n)
{
	hostBuffer.reserve(n);
}

template <class T>
void AbstractComputeBuffer<T>::Fit()
{
	FitHost();
	FitDevice();
}

template <class T>
void AbstractComputeBuffer<T>::FitHost()
{
	hostBuffer.shrink_to_fit();
}

template <typename T>
template <class ... Args>
void AbstractComputeBuffer<T>::EmplaceBack(Args&& ... args)
{
	hostBuffer.emplace_back(args...);
}

template <class T>
void AbstractComputeBuffer<T>::PushBack(const T& val)
{
	hostBuffer.push_back(val);
}

template <class T>
void AbstractComputeBuffer<T>::PushBack(T&& val)
{
	hostBuffer.push_back(val);
}

template <class T>
void AbstractComputeBuffer<T>::PopBack()
{
	hostBuffer.pop_back();
}

template <typename T>
template <class ... Args>
typename AbstractComputeBuffer<T>::iterator AbstractComputeBuffer<T>::Emplace(const_iterator itr, Args&& ... args)
{
	return hostBuffer.emplace(itr, args...);
}

template <class T>
typename AbstractComputeBuffer<T>::iterator AbstractComputeBuffer<T>::Insert(const_iterator itr, const T& val)
{
	return hostBuffer.insert(itr, val);
}

template <class T>
typename AbstractComputeBuffer<T>::iterator AbstractComputeBuffer<T>::Insert(const_iterator itr, T&& val)
{
	return hostBuffer.insert(itr, val);
}

template <class T>
typename AbstractComputeBuffer<T>::iterator AbstractComputeBuffer<T>::Insert(const_iterator itr, size_type n, const T& val)
{
	return hostBuffer.insert(itr, n, val);
}

template <typename T>
template <class InputIt>
typename AbstractComputeBuffer<T>::iterator AbstractComputeBuffer<T>::Insert(const_iterator itr, InputIt begin, InputIt end)
{
	return hostBuffer.insert(itr, begin, end);
}

template <class T>
typename AbstractComputeBuffer<T>::iterator AbstractComputeBuffer<T>::Erase(const_iterator itr)
{
	return hostBuffer.erase(itr);
}

template <class T>
typename AbstractComputeBuffer<T>::iterator AbstractComputeBuffer<T>::Erase(const_iterator begin, const_iterator end)
{
	return hostBuffer.erase(begin, end);
}

template <class T>
void AbstractComputeBuffer<T>::Swap(AbstractComputeBuffer<T>& other)
{
	hostBuffer.swap(other.hostBuffer);
	std::swap(flags, other.flags);
}

template <class T>
void AbstractComputeBuffer<T>::Clear() noexcept
{
	hostBuffer.clear();
}