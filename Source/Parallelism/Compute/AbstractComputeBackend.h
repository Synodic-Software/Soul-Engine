#pragma once

#include "ComputeDevice.h"
#include "ComputeBuffer.h"

class AbstractComputeBackend {

public:

	virtual ~AbstractComputeBackend() = default;


	virtual void ExtractDevices(std::vector<ComputeDevice>&) = 0;

	virtual void InitThread() = 0;

protected:

private:

};
