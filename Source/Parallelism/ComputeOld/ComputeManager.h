#pragma once

#include "CUDA/CUDABackend.h"
#include "OpenCL/OpenCLBackend.h"

#include <vector>

#define S_BEST_DEVICE ComputeManager::Instance().GetBestGPU()

class ComputeManager {

public:

	static ComputeManager& Instance()
	{
		static ComputeManager instance;
		return instance;
	}

	ComputeManager(ComputeManager const&) = delete;
	void operator=(ComputeManager const&) = delete;


	void ExtractDevices();
	void DestroyDevices();

	void InitThread();

	ComputeDevice GetBestGPU();
	ComputeDevice GetBestCPU();


private:

	ComputeManager() = default;

	std::vector<ComputeDevice> devices;

	CUDABackend cudaBackend;
	OpenCLBackend openCLBackend;
};
