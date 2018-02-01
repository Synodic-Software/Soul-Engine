#include "ComputeManager.h"

void ComputeManager::ExtractDevices() {

	cudaBackend.ExtractDevices(devices);
	openCLBackend.ExtractDevices(devices);

}

void ComputeManager::DestroyDevices() {
	devices.clear();
}

void ComputeManager::InitThread() {
	cudaBackend.InitThread();
	openCLBackend.InitThread();
}

ComputeDevice ComputeManager::GetBestGPU() {
	return devices[0];
}

ComputeDevice ComputeManager::GetBestCPU() {
	return devices[0];
}

