#pragma once

#include "glm/glm.hpp"

#include "CUDA/CUDADevice.cuh"
#include "OpenCL/OpenCLDevice.h"

#include "GPGPU/GPUExecutePolicy.h"

/* A GPU device. */
class GPUDevice {

public:

	/*
	 *    Constructor.
	 *    @param	parameter1	The first parameter.
	 */

	GPUDevice(GPUDeviceBase*);

	/* Destructor. */
	virtual ~GPUDevice();

	template <typename KernelFunction, typename... Args>
	void Launch(const GPUExecutePolicy& policy,
		const KernelFunction& kernel,
		Args ... parameters) {

		if (device->api == CUDA) {

			auto cudaDevice = static_cast<CUDADevice*>(device);
			cudaDevice->Launch(policy, kernel, parameters...);

		}
		else {
			//auto openCLDevice = static_cast<OpenCLDevice*>(device);
			//TODO implement
			//openCLDevice->Launch(policy, kernel, parameters...);
		}
		 
	}

	/*
	*    Gets core count.
	*    @return	The core count.
	*/

	virtual int GetCoreCount();


	/*
	*    Gets warp size.
	*    @return	The warp size.
	*/

	virtual int GetWarpSize();

	/*
	*    Gets sm count.
	*    @return	The sm count.
	*/

	virtual int GetSMCount();

	/*
	*    Gets warps per mp.
	*    @return	The warps per mp.
	*/

	virtual int GetWarpsPerMP();

	/*
	*    Gets blocks per mp.
	*    @return	The blocks per mp.
	*/

	virtual int GetBlocksPerMP();

	template<typename KernelFunction>
	GPUExecutePolicy BestExecutePolicy(const KernelFunction& kernel)
	{
		if (device->api == CUDA) {
			auto cudaDevice = static_cast<CUDADevice*>(device);
			return cudaDevice->BestExecutePolicy(kernel, [](int blockSize) -> int
			{
				return blockSize / 32 * sizeof(int);
			}
			);
		}
		else {


		}
	}

	GPUBackend GetAPI() const {
		return device->api;
	}

	int GetOrder() const {
		return device->order;
	}

protected:

	GPUDeviceBase* device;

private:

};