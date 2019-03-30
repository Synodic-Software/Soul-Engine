#include "ComputeBackend.h"

#include "Modules/CPU/CPUBackend.h"
#include "Modules/CUDA/CUDABackend.h"
#include "Modules/OpenCL/OpenCLBackend.h"
#include "Modules/Mock/MockBackend.h"


std::shared_ptr<ComputeBackend> ComputeBackend::CreateModule() {

    auto backend = std::make_shared<ComputeBackend>();

	backend->modules.reserve(3);
	backend->modules.push_back(std::make_unique<CPUBackend>());
	backend->modules.push_back(std::make_unique<CUDABackend>());
	backend->modules.push_back(std::make_unique<OpenCLBackend>());

	return backend;

}
