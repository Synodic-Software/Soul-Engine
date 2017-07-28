#pragma once

#include <vector>

/* . */
namespace CUDABackend {

	/*
	 *    Extracts the devices described by parameter1.
	 *    @param [in,out]	parameter1	The first parameter.
	 */

	void ExtractDevices(std::vector<int>&);

	/* Initializes the thread. */
	void InitThread();

	/*
	 *    Gets core count.
	 *    @return	The core count.
	 */

	int GetCoreCount();

	/*
	 *    Gets warp size.
	 *    @return	The warp size.
	 */

	int GetWarpSize();

	/*
	 *    Gets sm count.
	 *    @return	The sm count.
	 */

	int GetSMCount();

	/*
	 *    Gets warps per mp.
	 *    @return	The warps per mp.
	 */

	int GetWarpsPerMP();

	/*
	 *    Gets blocks per mp.
	 *    @return	The blocks per mp.
	 */

	int GetBlocksPerMP();

	/* Terminates this object. */
	void Terminate();

}