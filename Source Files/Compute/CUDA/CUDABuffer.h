#pragma once
#include "Compute\DeviceBuffer.h"

#include "Metrics.h"
#include "Utility/CUDA/CudaHelper.cuh"

/* Buffer for cuda. */
template<class T>
class CUDABuffer :public virtual DeviceBuffer<T> {

public:

	//Types

	typedef T                                     value_type;
	typedef uint		                          size_type;


	//Construction and Destruction 

	CUDABuffer(const GPUDevice& _device);

	~CUDABuffer();

	//Data Migration

	void Move(const GPUDevice&) override;

	void TransferToHost(std::vector<T>&) override;
	void TransferToDevice(std::vector<T>&) override;


	//Element Access

	T* Data() override;
	const T* Data() const override;


	//Capacity

	bool Empty() const noexcept override;

	size_type Size() const noexcept override;

	size_type MaxSize() const noexcept override;

	size_type Capacity() const noexcept override;

	void Resize(size_type) override;
	void Resize(size_type, const T&) override;

	void Reserve(size_type) override;

	void Fit() override;


private:

	T* buffer;

	void Reallocate() override;

};

template <class T>
CUDABuffer<T>::CUDABuffer(const GPUDevice& device):
	DeviceBuffer(device),
	buffer(nullptr)
{

}

template <class T>
CUDABuffer<T>::~CUDABuffer() {
	CudaCheck(cudaFree(buffer));
	buffer = nullptr;
}

template <class T>
void CUDABuffer<T>::Move(const GPUDevice&)
{
	//TODO implement
}

template <class T>
void CUDABuffer<T>::TransferToHost(std::vector<T>& hostBuffer) {

	//perform size checks
	if (DeviceBuffer<T>::size > hostBuffer.size()) {
		hostBuffer.resize(DeviceBuffer<T>::size);
	}

	CudaCheck(cudaMemcpy(hostBuffer.data(), buffer, DeviceBuffer<T>::size * sizeof(T), cudaMemcpyDeviceToHost));

}

template <class T>
void CUDABuffer<T>::TransferToDevice(std::vector<T>& hostBuffer) {

	//perform size checks
	if (hostBuffer.size() > DeviceBuffer<T>::size) {
		Resize(static_cast<unsigned int>(hostBuffer.size()));
	}
	
	CudaCheck(cudaMemcpy(buffer, hostBuffer.data(), DeviceBuffer<T>::size * sizeof(T), cudaMemcpyHostToDevice));

}

template <class T>
T* CUDABuffer<T>::Data() {
	//TODO implement
	return nullptr;
}

template <class T>
const T* CUDABuffer<T>::Data() const {
	//TODO implement
	return nullptr;
}

template <class T>
bool CUDABuffer<T>::Empty() const noexcept {
	//TODO implement
	return true;
}

template <class T>
typename CUDABuffer<T>::size_type CUDABuffer<T>::Size() const noexcept {
	//TODO implement
	return 0;
}

template <class T>
typename CUDABuffer<T>::size_type CUDABuffer<T>::MaxSize() const noexcept {
	//TODO implement
	return 0;
}

template <class T>
typename CUDABuffer<T>::size_type CUDABuffer<T>::Capacity() const noexcept {
	//TODO implement
	return 0;
}

template <class T>
void CUDABuffer<T>::Resize(size_type n) {

	if (n > DeviceBuffer<T>::size) {
		Reserve(n);
	}

	DeviceBuffer<T>::size = n;

}

template <class T>
void CUDABuffer<T>::Resize(size_type, const T&) {
	//TODO implement
}

template <class T>
void CUDABuffer<T>::Reserve(size_type n) {

	if (n > DeviceBuffer<T>::capacity) {
		DeviceBuffer<T>::capacity = n;
		Reallocate();
	}

}

template <class T>
void CUDABuffer<T>::Fit()
{
	//TODO implement
}

template <typename T>
void CUDABuffer<T>::Reallocate() {

	T* temp;
	CudaCheck(cudaMalloc((void**)&temp, DeviceBuffer<T>::capacity * sizeof(T)));

	CudaCheck(cudaMemcpy(temp, buffer, DeviceBuffer<T>::size * sizeof(T), cudaMemcpyDeviceToDevice));
	
	CudaCheck(cudaFree(buffer));
	buffer = temp;

}