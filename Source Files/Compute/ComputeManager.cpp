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
	//TODO implement
	return devices[0];
}

ComputeDevice ComputeManager::GetBestCPU() {
	//TODO implement
	return devices.back();
}

