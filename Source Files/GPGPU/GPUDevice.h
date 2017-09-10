#pragma once

#include "Metrics.h"

/* Values that represent GPU backends. */
enum GPUBackend { CUDA, OpenCL };

/* A GPU device. */
class GPUDevice {

public:

	/*
	 *    Constructor.
	 *    @param	parameter1	The first parameter.
	 */

	GPUDevice(uint);
	/* Destructor. */
	virtual ~GPUDevice();

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