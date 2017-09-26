#pragma once
#include "GPGPU\GPUBufferBase.h"

#include "Metrics.h"
#include "GPGPU\CUDA\CUDADevice.h"
#include "Utility/CUDA/CudaHelper.cuh"

/* Buffer for cuda. */
template<class T>
class CUDABuffer :public GPUBufferBase<T> {

public:

	CUDABuffer(const GPUDevice& _device, uint _byteCount)
		: GPUBufferBase(_device, _byteCount) {
		CudaCheck(cudaMalloc((void**)&deviceData, device_size * sizeof(T)));
	}

	CUDABuffer(const GPUDevice& _device, GPUBufferBase<T>& other)
		: GPUBufferBase(_device, other) {

		device_size = other.size();
		device_capacity = other.capacity();

		//allocate the new space
		CudaCheck(cudaMalloc((void**)&deviceData, host_capacity * sizeof(T)));
		CudaCheck(cudaMemcpy(deviceData, other.device_data(), device_size * sizeof(T), cudaMemcpyDeviceToDevice));

	}

	~CUDABuffer() {
		if (deviceData) {
			CudaCheck(cudaFree(deviceData));
		}
	}


	void TransferToHost() override {

		//perform size checks
		if (device_size > host_size) {
			GPUBufferBase<T>::resize(device_size);
		}

		CudaCheck(cudaMemcpy(hostData, deviceData, device_size * sizeof(T), cudaMemcpyDeviceToHost));
	}

	void TransferToDevice() override {

		//perform size checks
		if (host_size > device_size) {
			CUDABuffer::resize(host_size);
		}

		CudaCheck(cudaMemcpy(deviceData, hostData, device_size * sizeof(T), cudaMemcpyHostToDevice));
	}

	void reserve(uint newCapacity) override {

		GPUBufferBase<T>::reserve(newCapacity);

		if (newCapacity > device_capacity) {
			//allocate the new size
			T* data;
			CudaCheck(cudaMalloc((void**)&data, newCapacity * sizeof(T)));

			uint lSize = newCapacity < device_size ? newCapacity : device_size;

			//move all previous data
			CudaCheck(cudaMemcpy(data, &deviceData, lSize * sizeof(T), cudaMemcpyDeviceToDevice));

			device_capacity = newCapacity;

			//deallocate formerly used memory
			if (deviceData) {
				CudaCheck(cudaFree(deviceData));
			}

			deviceData = data; //reassign the data pointer
		}
	}

	void resize(uint newSize) override {

		//first resize the host
		GPUBufferBase<T>::resize(newSize);

		reserve(newSize);
		device_size = newSize;  //change host size.

	}


	GPUBufferBase<T>& operator=(GPUBufferBase<T>& rhs) override {
		return GPUBufferBase<T>::operator=(rhs);
	}

protected:

private:

};