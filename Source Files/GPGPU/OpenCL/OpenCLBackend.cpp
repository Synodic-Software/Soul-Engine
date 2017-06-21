#include "OpenCLBackend.h"

#include "Utility\Logger.h"

/* Number of devices */
/* Number of devices */
static int deviceCount;
//cudaDeviceProp* deviceProp;

namespace OpenCLBackend {

	/*
	 *    Extracts the devices described by devices.
	 *
	 *    @param [in,out]	devices	The devices.
	 */

	void ExtractDevices(std::vector<int>& devices) {

		S_LOG_WARNING("OpenCL is not supported yet");
	}

	/* Initializes the thread. */
	/* Initializes the thread. */
	void InitThread() {

	}

	/*
	 *    Gets core count.
	 *
	 *    @return	The core count.
	 */

	int GetCoreCount() {

		return 0;
	}

	/*
	 *    Gets sm count.
	 *
	 *    @return	The sm count.
	 */

	int GetSMCount() {

		return 0;
	}

	/*
	 *    Gets warp size.
	 *
	 *    @return	The warp size.
	 */

	int GetWarpSize() {
		
		return 0;
	}

	/*
	 *    Gets block height.
	 *
	 *    @return	The block height.
	 */

	int GetBlockHeight() {
		
		return 0;
	}

	/* Terminates this object. */
	/* Terminates this object. */
	void Terminate() {
		
	}

}