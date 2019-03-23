#pragma once
#include "Parallelism/Compute/DeviceBuffer.h"

#include "Types.h"
#include "Parallelism/Compute/ComputeDevice.h"

/* Buffer for open cl. */
template<class T>
class OpenCLBuffer :public virtual DeviceBuffer<T> {

public:

	//Types

	typedef T                                     value_type;
	typedef uint		                          size_type;


	//Construction and Destruction 

	OpenCLBuffer(const ComputeDevice& _device);

	~OpenCLBuffer();

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

private:

	void Reallocate() override;

};

template <class T>
OpenCLBuffer<T>::OpenCLBuffer(const ComputeDevice& device) :
	DeviceBuffer<T>(device)
{

}

template <class T>
OpenCLBuffer<T>::~OpenCLBuffer() {
	//TODO implement
}

template <class T>
void OpenCLBuffer<T>::Move(const ComputeDevice&)
{
	//TODO implement
}

template <class T>
void OpenCLBuffer<T>::TransferToHost(std::vector<T>& hostBuffer) {
	//TODO implement
}

template <class T>
void OpenCLBuffer<T>::TransferToDevice(std::vector<T>& hostBuffer) {
	//TODO implement
}

template <class T>
T* OpenCLBuffer<T>::Data() {
	//TODO implement
	return nullptr;
}

template <class T>
const T* OpenCLBuffer<T>::Data() const {
	//TODO implement
	return nullptr;
}

template <class T>
bool OpenCLBuffer<T>::Empty() const noexcept {
	//TODO implement
	return true;
}

template <class T>
typename OpenCLBuffer<T>::size_type OpenCLBuffer<T>::MaxSize() const noexcept {
	//TODO implement
	return 0;
}

template <class T>
void OpenCLBuffer<T>::Resize(size_type n) {

	if (n > DeviceBuffer<T>::size) {
		Reserve(n);
	}

	DeviceBuffer<T>::size = n;

}

template <class T>
void OpenCLBuffer<T>::Resize(size_type, const T&) {
	//TODO implement
}

template <class T>
void OpenCLBuffer<T>::Reserve(size_type n) {

	if (n > DeviceBuffer<T>::capacity) {
		DeviceBuffer<T>::capacity = n;
		Reallocate();
	}

}

template <class T>
void OpenCLBuffer<T>::Fit()
{
	//TODO implement
}

template <typename T>
void OpenCLBuffer<T>::Reallocate() {
	//TODO implement
}
