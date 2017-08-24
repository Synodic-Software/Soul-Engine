#include "OpenCLBackend.h"

#include "Utility\Logger.h"

/* Number of devices */
static int deviceCount;


	/*
	 *    Extracts the devices described by devices.
	 *    @param [in,out]	devices	The devices.
	 */

void OpenCLBackend::ExtractDevices(std::vector<GPUDevice>& devices) {

	S_LOG_WARNING("OpenCL is not supported yet");
}

/* Initializes the thread. */
void OpenCLBackend::InitThread() {

}

/*
 *    Gets core count.
 *    @return	The core count.
 */

int OpenCLBackend::GetCoreCount() {

	return 0;
}

/*
 *    Gets sm count.
 *    @return	The sm count.
 */

int OpenCLBackend::GetSMCount() {

	return 0;
}

/*
 *    Gets warp size.
 *    @return	The warp size.
 */

int OpenCLBackend::GetWarpSize() {

	return 0;
}

/*
 *    Gets block height.
 *    @return	The block height.
 */

int OpenCLBackend::GetBlockHeight() {

	return 0;
}

/* Terminates this object. */
void OpenCLBackend::Terminate() {

}
