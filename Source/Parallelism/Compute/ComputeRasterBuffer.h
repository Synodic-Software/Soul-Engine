#pragma once

#include "AbstractComputeBuffer.h"
#include "Parallelism/Compute/ComputeDevice.h"

#include "Parallelism/Compute/CUDA/CUDARasterBuffer.h"
#include "Parallelism/Compute/OpenCL/OpenCLRasterBuffer.h"

/* Buffer for GPU raster. */
template <class T>
class ComputeRasterBuffer : public AbstractComputeBuffer<T>{

public:

	//Types
	
	typedef std::unique_ptr<DeviceRasterBuffer<T>>                    device_pointer;
	typedef typename AbstractComputeBuffer<T>::value_type             value_type;
	typedef typename AbstractComputeBuffer<T>::reference              reference;
	typedef typename AbstractComputeBuffer<T>::const_reference        const_reference;
	typedef typename AbstractComputeBuffer<T>::pointer                pointer;
	typedef typename AbstractComputeBuffer<T>::const_pointer          const_pointer;
	typedef typename AbstractComputeBuffer<T>::iterator               iterator;
	typedef typename AbstractComputeBuffer<T>::const_iterator         const_iterator;
	typedef typename AbstractComputeBuffer<T>::reverse_iterator       reverse_iterator;
	typedef typename AbstractComputeBuffer<T>::const_reverse_iterator const_reverse_iterator;
	typedef typename AbstractComputeBuffer<T>::difference_type        difference_type;
	typedef typename AbstractComputeBuffer<T>::size_type		      size_type;


	//Construction and Destruction 

	ComputeRasterBuffer();
	ComputeRasterBuffer(const ComputeDevice&);
	ComputeRasterBuffer(const ComputeDevice&, size_type);
	ComputeRasterBuffer(const ComputeDevice&, size_type, const T&);
	ComputeRasterBuffer(const ComputeRasterBuffer&);
	ComputeRasterBuffer(ComputeRasterBuffer&&) noexcept;

	~ComputeRasterBuffer() = default;

	ComputeRasterBuffer<T>& operator= (ComputeRasterBuffer<T>&&) noexcept;
	ComputeRasterBuffer<T>& operator= (const ComputeRasterBuffer<T>&);

	T* DataDevice() override;
	const T* DataDevice() const override;


	//Data Migration

	void Move(const ComputeDevice&) override;

	void TransferToHost() override;
	void TransferToDevice() override;

	//Capacity

	bool EmptyDevice() const noexcept override;

	size_type SizeDevice() const noexcept override;

	size_type MaxSizeDevice() const noexcept override;

	size_type DeviceCapacity() const noexcept override;

	void ResizeDevice(size_type) override;
	void ResizeDevice(size_type, const T&) override;

	void ReserveDevice(size_type) override;

	void FitDevice() override;

	void Swap(ComputeRasterBuffer<T>&);

	//Raster Functions

	void MapResources();

	void UnmapResources();

	void BindData(uint pos);


private:

	device_pointer buffer;

};

template <class T>
ComputeRasterBuffer<T>::ComputeRasterBuffer() :
	AbstractComputeBuffer<T>()
{
}

template <class T>
ComputeRasterBuffer<T>::ComputeRasterBuffer(const ComputeDevice& device) :
	AbstractComputeBuffer<T>(device)
{

	if (device.GetBackend() == CUDA_API) {
		buffer.reset(new CUDARasterBuffer<T>(device));
	}
	else if (device.GetBackend() == OPENCL_API) {
		buffer.reset(new OpenCLRasterBuffer<T>(device));
	}

}

template <class T>
ComputeRasterBuffer<T>::ComputeRasterBuffer(const ComputeDevice& device, size_type n) :
	AbstractComputeBuffer<T>(device, n)
{

	if (device.GetBackend() == CUDA_API) {
		buffer.reset(new CUDARasterBuffer<T>(device, n));
	}
	else if (device.GetBackend() == OPENCL_API) {
		buffer.reset(new OpenCLRasterBuffer<T>(device, n));
	}

}

template <class T>
ComputeRasterBuffer<T>::ComputeRasterBuffer(const ComputeDevice& device, size_type n, const T &val) :
	AbstractComputeBuffer<T>(device, n, val)
{
	if (device.GetBackend() == CUDA_API) {
		buffer.reset(new CUDARasterBuffer<T>(device, n, val));
	}
	else if (device.GetBackend() == OPENCL_API) {
		buffer.reset(new OpenCLRasterBuffer<T>(device, n, val));
	}
}


template <class T>
ComputeRasterBuffer<T>::ComputeRasterBuffer(const ComputeRasterBuffer& other) :
	AbstractComputeBuffer<T>(other)
{
	*this = other;
}

template <class T>
ComputeRasterBuffer<T>::ComputeRasterBuffer(ComputeRasterBuffer<T>&& other) noexcept :
	AbstractComputeBuffer<T>(other)
{
	*this = std::move(other);
}

template <class T>
ComputeRasterBuffer<T>& ComputeRasterBuffer<T>::operator= (ComputeRasterBuffer<T>&& other) noexcept
{
	AbstractComputeBuffer<T>::operator=(other);
	buffer = std::move(other.buffer);

	return *this;
}

template <class T>
ComputeRasterBuffer<T>& ComputeRasterBuffer<T>::operator= (const ComputeRasterBuffer<T>& other)
{

	AbstractComputeBuffer<T>::operator=(other);

	if (other.buffer->GetBackend() == CUDA_API) {
		CUDARasterBuffer<T> temp = *static_cast<CUDARasterBuffer<T>*>(other.buffer.get());
		buffer.reset(new CUDARasterBuffer<T>(temp));
	}
	else if (other.buffer->GetBackend() == OPENCL_API)
	{
		OpenCLRasterBuffer<T> temp = *static_cast<OpenCLRasterBuffer<T>*>(other.buffer.get());
		buffer.reset(new OpenCLRasterBuffer<T>(temp));
	}

	return *this;

}

/*
* Move the buffer to the specified device
*
* @tparam	T	Generic type parameter.
* @param	devices	The devices to transfer to.
*/
template <class T>
void ComputeRasterBuffer<T>::Move(const ComputeDevice& device) {

	if (buffer) {
		buffer->Move(device);
	}
	else
	{
		if (device.GetBackend() == CUDA_API) {
			buffer.reset(new CUDARasterBuffer<T>(device));
		}
		else if (device.GetBackend() == OPENCL_API)
		{
			buffer.reset(new OpenCLRasterBuffer<T>(device));
		}
	}

}

template <class T>
void ComputeRasterBuffer<T>::TransferToHost()
{
	buffer->TransferToHost(AbstractComputeBuffer<T>::hostBuffer);
}

template <class T>
void ComputeRasterBuffer<T>::TransferToDevice()
{
	buffer->TransferToDevice(AbstractComputeBuffer<T>::hostBuffer);
}

template <class T>
T* ComputeRasterBuffer<T>::DataDevice() {
	return buffer->Data();
}

template <class T>
const T* ComputeRasterBuffer<T>::DataDevice() const {
	return buffer->Data();
}

template <class T>
bool ComputeRasterBuffer<T>::EmptyDevice() const noexcept
{
	return buffer->Empty();
}

template <class T>
typename ComputeRasterBuffer<T>::size_type ComputeRasterBuffer<T>::SizeDevice() const noexcept
{
	return buffer->Size();
}

template <class T>
typename ComputeRasterBuffer<T>::size_type ComputeRasterBuffer<T>::DeviceCapacity() const noexcept
{
	return buffer->Capacity();
}

template <class T>
typename ComputeRasterBuffer<T>::size_type ComputeRasterBuffer<T>::MaxSizeDevice() const noexcept
{
	return buffer->MaxSize();
}


template <class T>
void ComputeRasterBuffer<T>::ResizeDevice(size_type n)
{
	buffer->Resize(n);
}

template <class T>
void ComputeRasterBuffer<T>::ResizeDevice(size_type n, const T& val)
{
	buffer->Resize(n, val);
}

template <class T>
void ComputeRasterBuffer<T>::ReserveDevice(size_type n)
{
	buffer->Reserve(n);
}

template <class T>
void ComputeRasterBuffer<T>::FitDevice()
{
	buffer->Fit();
}

template <class T>
void ComputeRasterBuffer<T>::Swap(ComputeRasterBuffer<T>& other)
{
	AbstractComputeBuffer<T>::Swap(other);
	buffer.swap(other.buffer);
}

template <class T>
void ComputeRasterBuffer<T>::MapResources() {
	buffer->MapResources();
}

template <class T>
void ComputeRasterBuffer<T>::UnmapResources() {
	buffer->UnmapResources();
}

template <class T>
void ComputeRasterBuffer<T>::BindData(uint pos) {
	buffer->BindData(pos);
}
