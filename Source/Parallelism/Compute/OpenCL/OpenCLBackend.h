#pragma once

#include "Parallelism/Compute/AbstractComputeBackend.h"

#include <vector>

/* . */
class OpenCLBackend: public AbstractComputeBackend {

public:

	~OpenCLBackend() = default;

	void ExtractDevices(std::vector<ComputeDevice>&) override;

	void InitThread() override;

private:

};