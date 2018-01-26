#pragma once

#include "Metrics.h"

/* Values that represent GPU backends. */
enum GPUBackend { CUDA_API, OPENCL_API };


/* A GPU device. */
class GPUDeviceBase {

public:

	/*
	 *    Constructor.
	 *    @param	parameter1	The first parameter.
	 */

	GPUDeviceBase(uint);

	/* Destructor. */
	virtual ~GPUDeviceBase();

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

	/* The API */
	GPUBackend api;

	/* The order */
	int order;
protected:

private:

};