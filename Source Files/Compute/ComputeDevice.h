#pragma once

#include "CUDA/CUDADevice.cuh"
#include "OpenCL/OpenCLDevice.h"

#include "Compute/GPUExecutePolicy.h"

/* A GPU device. */
class ComputeDevice {

public:

	ComputeDevice(ComputeBackend, int, int);

	int GetCoreCount() const;
	int GetWarpSize() const;
	int GetSMCount() const;
	int GetWarpsPerMP() const;
	int GetBlocksPerMP() const;

	ComputeBackend GetBackend() const;
	int GetOrder() const;

	template <typename KernelFunction, typename... Args>
	void Launch(const GPUExecutePolicy& policy,
		uint,
		const KernelFunction& kernel,
		Args&& ... parameters);

	template <typename KernelFunction, typename... Args>
	void LaunchOld(const GPUExecutePolicy& policy,
		const KernelFunction& kernel,
		Args&& ... parameters);

	template<typename KernelFunction>
	GPUExecutePolicy BestExecutePolicy(const KernelFunction& kernel);

protected:

	std::shared_ptr<AbstractComputeDevice> device;

	ComputeBackend backend;
	int order;

};


template <typename KernelFunction, typename... Args>
void ComputeDevice::Launch(const GPUExecutePolicy& policy,
	uint n,
	const KernelFunction& kernel,
	Args&& ... parameters) {

	if (backend == CUDA_API) {

		auto cudaDevice = static_cast<CUDADevice*>(device.get());
		cudaDevice->Launch(policy,n, kernel, std::forward<Args>(parameters)...);

	}
	else {
		//auto openCLDevice = static_cast<OpenCLDevice*>(device);
		//TODO implement
		//openCLDevice->Launch(policy, kernel, std::forward<Args>(parameters)...);
	}

}

template <typename KernelFunction, typename... Args>
void ComputeDevice::LaunchOld(const GPUExecutePolicy& policy,
	const KernelFunction& kernel,
	Args&& ... parameters) {

	if (backend == CUDA_API) {

		auto cudaDevice = static_cast<CUDADevice*>(device.get());
		cudaDevice->LaunchOld(policy, kernel, std::forward<Args>(parameters)...);

	}
	else {
		//auto openCLDevice = static_cast<OpenCLDevice*>(device);
		//TODO implement
		//openCLDevice->Launch(policy, kernel, std::forward<Args>(parameters)...);
	}

}

template<typename KernelFunction>
GPUExecutePolicy ComputeDevice::BestExecutePolicy(const KernelFunction& kernel)
{
	if (GetBackend() == CUDA_API) {
		auto cudaDevice = static_cast<CUDADevice*>(device.get());
		return cudaDevice->BestExecutePolicy(kernel, [](int blockSize) -> int
		{
			return blockSize / 32 * sizeof(int);
		}
		);
	}
	else {

		//TODO implement

	}
}