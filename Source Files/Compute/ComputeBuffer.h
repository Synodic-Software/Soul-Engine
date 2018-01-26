#pragma once

/*
*    The wrapper class which passes all functions to the actual buffer,
*    a user friendly factory class encapsulating the polymorphic buffer
*/


#include "Compute/AbstractComputeBuffer.h"

#include "Compute/GPUDevice.h"
#include "Compute/DeviceBuffer.h"

#include "Compute/CUDA/CUDABuffer.h"
#include "Compute/OpenCL/OpenCLBuffer.h"

#include "Utility/Logger.h"

/*
 *    Buffer for device/host(GPU/CPU) communication and storage.
 *    @tparam	T	Generic type parameter.
 */

 //TODO add initializer list functions

template <class T>
class ComputeBuffer: public AbstractComputeBuffer<T> {

public:

	//Types

	typedef std::unique_ptr<DeviceBuffer<T>>                          device_pointer;
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

	ComputeBuffer();
	ComputeBuffer(const GPUDevice&);
	ComputeBuffer(const GPUDevice&, size_type);
	ComputeBuffer(const GPUDevice&, size_type, const T&);
	ComputeBuffer(const ComputeBuffer&);
	ComputeBuffer(ComputeBuffer&&) noexcept;

	~ComputeBuffer() = default;

	ComputeBuffer<T>& operator= (ComputeBuffer<T>&&) noexcept;
	ComputeBuffer<T>& operator= (const ComputeBuffer<T>&);

	T* DataDevice() override;
	const T* DataDevice() const override;


	//Data Migration

	void Move(const GPUDevice&) override;

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

	void Swap(ComputeBuffer<T>&);

private:

	device_pointer deviceBuffer;

};

template <class T>
ComputeBuffer<T>::ComputeBuffer() :
	AbstractComputeBuffer()
{
}

template <class T>
ComputeBuffer<T>::ComputeBuffer(const GPUDevice& device) :
	AbstractComputeBuffer(device)
{

	if (device.GetAPI() == CUDA_API) {
		deviceBuffer.reset(new CUDABuffer<T>(device));
	}
	else if (device.GetAPI() == OPENCL_API) {
		deviceBuffer.reset(new OpenCLBuffer<T>(device));
	}

}

template <class T>
ComputeBuffer<T>::ComputeBuffer(const GPUDevice& device, size_type n) :
	AbstractComputeBuffer(device, n)
{

	if (device.GetAPI() == CUDA_API) {
		deviceBuffer.reset(new CUDABuffer<T>(device, n));
	}
	else if (device.GetAPI() == OPENCL_API) {
		deviceBuffer.reset(new OpenCLBuffer<T>(device, n));
	}

}

template <class T>
ComputeBuffer<T>::ComputeBuffer(const GPUDevice& device, size_type n, const T &val) :
	AbstractComputeBuffer(device, n, val)
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
	AbstractComputeBuffer<T>::operator=(other);
	deviceBuffer = std::move(other.deviceBuffer);

	return *this;
}

template <class T>
ComputeBuffer<T>& ComputeBuffer<T>::operator= (const ComputeBuffer<T>& other)
{

	AbstractComputeBuffer<T>::operator=(other);

	if (other.deviceBuffer->GetAPI() == CUDA_API) {
		CUDABuffer<T> temp = *dynamic_cast<CUDABuffer<T>*>(other.deviceBuffer.get());
		deviceBuffer.reset(new CUDABuffer<T>(temp));
	}
	else if (other.deviceBuffer->GetAPI() == OPENCL_API)
	{
		OpenCLBuffer<T> temp = *dynamic_cast<OpenCLBuffer<T>*>(other.deviceBuffer.get());
		deviceBuffer.reset(new OpenCLBuffer<T>(temp));
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
void ComputeBuffer<T>::Move(const GPUDevice& device) {

	if (deviceBuffer) {
		deviceBuffer->Move(device);
	}
	else
	{
		if (device.GetAPI() == CUDA_API) {
			deviceBuffer.reset(new CUDABuffer<T>(device));
		}
		else if (device.GetAPI() == OPENCL_API)
		{
			deviceBuffer.reset(new OpenCLBuffer<T>(device));
		}
	}

}

template <class T>
void ComputeBuffer<T>::TransferToHost()
{
	deviceBuffer->TransferToHost(AbstractComputeBuffer<T>::hostBuffer);
}

template <class T>
void ComputeBuffer<T>::TransferToDevice()
{
	deviceBuffer->TransferToDevice(AbstractComputeBuffer<T>::hostBuffer);
}

template <class T>
T* ComputeBuffer<T>::DataDevice() {
	return deviceBuffer->Data();
}

template <class T>
const T* ComputeBuffer<T>::DataDevice() const {
	return deviceBuffer->Data();
}

template <class T>
bool ComputeBuffer<T>::EmptyDevice() const noexcept
{
	return deviceBuffer->Empty();
}

template <class T>
typename ComputeBuffer<T>::size_type ComputeBuffer<T>::SizeDevice() const noexcept
{
	return deviceBuffer->Size();
}

template <class T>
typename ComputeBuffer<T>::size_type ComputeBuffer<T>::DeviceCapacity() const noexcept
{
	return deviceBuffer->Capacity();
}

template <class T>
typename ComputeBuffer<T>::size_type ComputeBuffer<T>::MaxSizeDevice() const noexcept
{
	return deviceBuffer->MaxSize();
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
void ComputeBuffer<T>::ReserveDevice(size_type n)
{
	deviceBuffer->Reserve(n);
}

template <class T>
void ComputeBuffer<T>::FitDevice()
{
	deviceBuffer->Fit();
}

template <class T>
void ComputeBuffer<T>::Swap(ComputeBuffer<T>& other)
{
	AbstractComputeBuffer<T>::Swap(other);
	deviceBuffer.swap(other.deviceBuffer);
}