#pragma once

#include "Parallelism/ComputeOld/AbstractComputeBackend.h"

#include <vector>

/* . */
class OpenCLBackend: public AbstractComputeBackend {

public:

	~OpenCLBackend() = default;

	void ExtractDevices(std::vector<ComputeDevice>&) override;

	void InitThread() override;

private:

};