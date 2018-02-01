#pragma once

#include "Compute/GPUExecutePolicy.h"
#include "Compute/AbstractComputeDevice.h"
#include <cuda_runtime.h>
#include "Utility/CUDA/CUDAHelper.cuh"

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
	void Launch(const GPUExecutePolicy&, const KernelFunction&, Args&&...);
	GPUExecutePolicy BestExecutePolicy(const void*, int(*)(int));

	
private:

	cudaDeviceProp deviceProperties;
	int deviceID;

};

template<typename KernelFunction, typename... Args>
void CUDADevice::Launch(const GPUExecutePolicy& policy, const KernelFunction& kernel, Args&&... parameters) {

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