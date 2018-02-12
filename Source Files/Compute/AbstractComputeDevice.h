#pragma once

/* Values that represent Compute backends. */
enum ComputeBackend { CUDA_API, OPENCL_API};


class AbstractComputeDevice {

public:

	AbstractComputeDevice() = default;
	virtual ~AbstractComputeDevice() = default;

	virtual int GetCoreCount() = 0;
	virtual int GetWarpSize() = 0;
	virtual int GetSMCount() = 0;
	virtual int GetWarpsPerMP() = 0;
	virtual int GetBlocksPerMP() = 0;

};