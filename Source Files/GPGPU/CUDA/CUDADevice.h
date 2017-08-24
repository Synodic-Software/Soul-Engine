#pragma once
#include "GPGPU\GPUDevice.h"
#include <cuda_runtime_api.h>

/* A cuda device. */
class CUDADevice :public GPUDevice {

public:

	/*
	 *    Constructor.
	 *    @param	parameter1	The first parameter.
	 */

	CUDADevice(uint);
	/* Destructor. */
	~CUDADevice() override;

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

protected:

private:
	cudaDeviceProp deviceProperties;

};