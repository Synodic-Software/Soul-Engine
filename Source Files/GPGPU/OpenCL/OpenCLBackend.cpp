#include "OpenCLBackend.h"

#include "Utility\Logger.h"

static int deviceCount;
//cudaDeviceProp* deviceProp;

namespace OpenCLBackend {

	void ExtractDevices(std::vector<int>& devices) {

		S_LOG_WARNING("OpenCL is not supported yet");
	}

	int GetCoreCount() {

		return 0;
	}

	int GetSMCount() {

		return 0;
	}

	int GetWarpSize() {
		
		return 0;
	}

	int GetBlockHeight() {
		
		return 0;
	}

	void Terminate() {
		
	}

}