#pragma once

#include <vector>
#include "Parallelism/Compute/AbstractComputeBackend.h"

class CUDABackend : public AbstractComputeBackend {

public:

	~CUDABackend() = default;

	void ExtractDevices(std::vector<ComputeDevice>&) override;

	void InitThread() override;

};