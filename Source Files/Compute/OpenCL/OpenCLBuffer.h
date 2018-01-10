#pragma once
#include "Compute\DeviceBuffer.h"

#include "Metrics.h"
#include "Compute\GPUDevice.h"

/* Buffer for open cl. */
template<class T>
class OpenCLBuffer :public DeviceBuffer<T> {

public:

	//Types

	typedef T                                     value_type;
	typedef uint		                          size_type;


	//Construction and Destruction 

	OpenCLBuffer(const GPUDevice& _device);

	~OpenCLBuffer();

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

	void Reallocate() override;

};

template <class T>
OpenCLBuffer<T>::OpenCLBuffer(const GPUDevice& device) :
	DeviceBuffer(device)
{

}

template <class T>
OpenCLBuffer<T>::~OpenCLBuffer() {
	//TODO implement
	throw std::exception("Not yet implemented");
}

template <class T>
void OpenCLBuffer<T>::Move(const GPUDevice&)
{
	//TODO implement
	throw std::exception("Not yet implemented");
}

template <class T>
void OpenCLBuffer<T>::TransferToHost(std::vector<T>& hostBuffer) {

	//TODO implement
	throw std::exception("Not yet implemented");

}

template <class T>
void OpenCLBuffer<T>::TransferToDevice(std::vector<T>& hostBuffer) {

	//TODO implement
	throw std::exception("Not yet implemented");

}

template <class T>
T* OpenCLBuffer<T>::Data() {
	//TODO implement
	throw std::exception("Not yet implemented");
}

template <class T>
const T* OpenCLBuffer<T>::Data() const {
	//TODO implement
	throw std::exception("Not yet implemented");
}

template <class T>
bool OpenCLBuffer<T>::Empty() const noexcept {
	//TODO implement
	throw std::exception("Not yet implemented");
}

template <class T>
typename OpenCLBuffer<T>::size_type OpenCLBuffer<T>::Size() const noexcept {
	//TODO implement
	throw std::exception("Not yet implemented");
}

template <class T>
typename OpenCLBuffer<T>::size_type OpenCLBuffer<T>::MaxSize() const noexcept {
	//TODO implement
	throw std::exception("Not yet implemented");
}

template <class T>
typename OpenCLBuffer<T>::size_type OpenCLBuffer<T>::Capacity() const noexcept {
	//TODO implement
	throw std::exception("Not yet implemented");
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
	throw std::exception("Not yet implemented");
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
	throw std::exception("Not yet implemented");
}

template <typename T>
void OpenCLBuffer<T>::Reallocate() {

	//TODO implement
	throw std::exception("Not yet implemented");

}