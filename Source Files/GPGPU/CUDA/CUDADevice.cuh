#pragma once

#include "GPGPU/GPUExecutePolicy.h"
#include "GPGPU/GPUDeviceBase.h"
#include <cuda_runtime.h>
#include "Utility/CUDA/CUDAHelper.cuh"

/* A cuda device. */
class CUDADevice :public GPUDeviceBase {

public:

	/*
	 *    Constructor.
	 *    @param	parameter1	The first parameter.
	 */

	CUDADevice(uint);
	/* Destructor. */
	~CUDADevice() override;

	template<typename KernelFunction, typename... Args>
	void Launch(const GPUExecutePolicy& policy, const KernelFunction& kernel, Args&... parameters) {

		const auto grid = dim3(policy.gridsize.x, policy.gridsize.y, policy.gridsize.z);
		const auto block = dim3(policy.blocksize.x, policy.blocksize.y, policy.blocksize.z);

		void* args[] = { static_cast<void*>(&parameters)... };

		CudaCheck(cudaLaunchKernel(
			static_cast<const void*>(&kernel),
			grid,
			block,
			args, 
			policy.sharedMemory,
			cudaStream_t(policy.stream)
		));

		CudaCheck(cudaPeekAtLastError());
		CudaCheck(cudaDeviceSynchronize());
	}

	/*
	*    Gets core count.
	*    @return	The core count.
	*/

	int GetCoreCount() override;

	/*
	*    Gets warp size.
	*    @return	The warp size.
	*/

	int GetWarpSize() override;

	/*
	*    Gets sm count.
	*    @return	The sm count.
	*/

	int GetSMCount() override;

	/*
	*    Gets warps per mp.
	*    @return	The warps per mp.
	*/

	int GetWarpsPerMP() override;

	/*
	*    Gets blocks per mp.
	*    @return	The blocks per mp.
	*/

	int GetBlocksPerMP() override;

	GPUExecutePolicy BestExecutePolicy(const void* kernel, int(*func)(int));

	

protected:

private:
	cudaDeviceProp deviceProperties;

};