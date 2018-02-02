#pragma once

#include "Compute/GPUExecutePolicy.h"
#include "Compute/AbstractComputeDevice.h"
#include <cuda_runtime.h>
#include "Utility/CUDA/CUDAHelper.cuh"
#include "Compute\DeviceAPI.h"


namespace detail {

	template <typename KernelFunction, typename... Args>
	__global__ void Invoker(KernelFunction, uint, Args...);


	template <typename KernelFunction, typename... Args>
	__global__ void Invoker(KernelFunction fn, uint n, Args... parameters) {

		const uint index = ThreadIndex1D();

		if (index >= n) {
			return;
		}

		fn(index, n, parameters...);
	}

}

/* A cuda device. */
class CUDADevice :public AbstractComputeDevice {

public:

	CUDADevice(int);
	~CUDADevice() override;

	
	int GetCoreCount() override;
	int GetWarpSize() override;
	int GetSMCount() override;
	int GetWarpsPerMP() override;
	int GetBlocksPerMP() override;


	template<typename KernelFunction, typename... Args>
	void LaunchOld(const GPUExecutePolicy&, const KernelFunction&, Args&&...);

	template<typename DeviceFunction, typename... Args>
	void Launch(const GPUExecutePolicy&, uint, const DeviceFunction&, Args&&...);

	GPUExecutePolicy BestExecutePolicy(const void*, int(*)(int));	
	
private:

	

	cudaDeviceProp deviceProperties;
	int deviceID;

};


template<typename KernelFunction, typename... Args>
void CUDADevice::LaunchOld(const GPUExecutePolicy& policy, const KernelFunction& kernel, Args&&... parameters) {

	const auto grid = dim3(policy.gridsize.x, policy.gridsize.y, policy.gridsize.z);
	const auto block = dim3(policy.blocksize.x, policy.blocksize.y, policy.blocksize.z);

	void* args[] = { static_cast<void*>(&parameters)... };

	CudaCheck(cudaLaunchKernel(
		static_cast<const void*>(&kernel),
		grid,
		block,
		args,
		policy.sharedMemory //TODO update with policy stream
	));

	CudaCheck(cudaPeekAtLastError());
	CudaCheck(cudaDeviceSynchronize());
}

template<typename DeviceFunction, typename... Args>
void CUDADevice::Launch(const GPUExecutePolicy& policy, uint n,const DeviceFunction& fn, Args&&... parameters) {

	const auto grid = dim3(policy.gridsize.x, policy.gridsize.y, policy.gridsize.z);
	const auto block = dim3(policy.blocksize.x, policy.blocksize.y, policy.blocksize.z);

	detail::Invoker << <grid,block, policy.sharedMemory, cudaStream_t(policy.stream) >> > (fn, n,parameters...);

	CudaCheck(cudaPeekAtLastError());
	CudaCheck(cudaDeviceSynchronize());
}