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
	inline void Launch(const GPUExecutePolicy& policy, const KernelFunction& kernel, Args... parameters) {

		dim3 grid = dim3(policy.gridsize.x, policy.gridsize.y, policy.gridsize.z);
		dim3 block = dim3(policy.blocksize.x, policy.blocksize.y, policy.blocksize.z);

		void* args[] = { static_cast<void*>(&parameters)... };


		CudaCheck(cudaLaunchKernel(
			static_cast<const void*>(&kernel),
			grid,
			block,
			args, 
			policy.sharedMemory,
			cudaStream_t(policy.stream)
		));

		//kernel << < grid, block, policy.sharedMemory, cudaStream_t(policy.stream) >> > (parameters...);

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