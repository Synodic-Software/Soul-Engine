#pragma once

#include <vector>
#include <Metrics.h>
#include "GPGPU\OpenCL\OpenCLRasterBuffer.h"

/* . */
/* . */
namespace OpenCLBackend {

	/*
	 *    Extracts the devices described by parameter1.
	 *
	 *    @param [in,out]	parameter1	The first parameter.
	 */

	void ExtractDevices(std::vector<int>&);

	/* Initializes the thread. */
	/* Initializes the thread. */
	void InitThread();

	/*
	 *    Gets core count.
	 *
	 *    @return	The core count.
	 */

	int GetCoreCount();

	/*
	 *    Gets warp size.
	 *
	 *    @return	The warp size.
	 */

	int GetWarpSize();

	/*
	 *    Gets sm count.
	 *
	 *    @return	The sm count.
	 */

	int GetSMCount();

	/*
	 *    Gets block height.
	 *
	 *    @return	The block height.
	 */

	int GetBlockHeight();

	/* Terminates this object. */
	/* Terminates this object. */
	void Terminate();

}