#pragma once
#include "Parallelism/Compute/DeviceBuffer.h"

#include "Core/Utility/Types.h"
#include "Parallelism/Compute/CUDA/Utility/CudaHelper.cuh"

/* Buffer for cuda. */
template<class T>
class CUDABuffer :public virtual DeviceBuffer<T> {

public:

	//Types

	typedef T                                     value_type;
	typedef uint		                          size_type;


	//Construction and Destruction 

	CUDABuffer(const ComputeDevice&);
	CUDABuffer(CUDABuffer const&);
	~CUDABuffer();

	CUDABuffer<T>& operator=(CUDABuffer<T> const&);


	//Data Migration

	void Move(const ComputeDevice&) override;

	void TransferToHost(std::vector<T>&) override;
	void TransferToDevice(std::vector<T>&) override;


	//Element Access

	T* Data() override;
	const T* Data() const override;


	//Capacity

	bool Empty() const noexcept override;

	size_type MaxSize() const noexcept override;

	void Resize(size_type) override;
	void Resize(size_type, const T&) override;

	void Reserve(size_type) override;

	void Fit() override;

protected:

	T* buffer;

private:	

	void Reallocate() override;

};

template <class T>
CUDABuffer<T>::CUDABuffer(const ComputeDevice& device):
	DeviceBuffer(device),
	buffer(nullptr)
{

}

template <class T>
CUDABuffer<T>::CUDABuffer(CUDABuffer const& other) :
	DeviceBuffer(other)
{
	*this = other;
}

template <class T>
CUDABuffer<T>::~CUDABuffer() {
	CudaCheck(cudaFree(buffer));
	buffer = nullptr;
}

template <class T>
CUDABuffer<T>& CUDABuffer<T>::operator=(CUDABuffer<T> const& other) {
	DeviceBuffer<T>::operator=(other);

	Reallocate();

	CudaCheck(cudaMemcpy(buffer, other.buffer, DeviceBuffer<T>::size * sizeof(T), cudaMemcpyDeviceToDevice));

	return *this;
}

template <class T>
void CUDABuffer<T>::Move(const ComputeDevice& device)
{
	//TODO implement memory transfer
	S_LOG_FATAL("Not implemented");
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
	return buffer;
}

template <class T>
const T* CUDABuffer<T>::Data() const {
	return buffer;
}

template <class T>
bool CUDABuffer<T>::Empty() const noexcept {
	//TODO implement
	S_LOG_FATAL("Not implemented");
	return true;
}

template <class T>
typename CUDABuffer<T>::size_type CUDABuffer<T>::MaxSize() const noexcept {
	//TODO implement
	S_LOG_FATAL("Not implemented");
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
	S_LOG_FATAL("Not implemented");
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
	S_LOG_FATAL("Not implemented");
}

template <typename T>
void CUDABuffer<T>::Reallocate() {

	void* temp;
	CudaCheck(cudaMalloc(&temp, DeviceBuffer<T>::capacity * sizeof(T)));

	if (buffer) {
		CudaCheck(cudaMemcpy(temp, buffer, DeviceBuffer<T>::size * sizeof(T), cudaMemcpyDeviceToDevice));
		CudaCheck(cudaFree(buffer));
	}

	buffer = static_cast<T*>(temp);

}