#pragma once
#include "Compute\GPUDevice.h"

/* An open cl device. */
class OpenCLDevice :public GPUDeviceBase {

public:

	/*
	 *    Constructor.
	 *    @param	parameter1	The first parameter.
	 */

	OpenCLDevice(uint);
	/* Destructor. */
	~OpenCLDevice() override;

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

};