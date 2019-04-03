#pragma once

#include "Parallelism/Compute/ComputeDevice.h"

class OpenCLDevice: public ComputeDevice{

public:

    OpenCLDevice() = default;
	virtual ~OpenCLDevice() = default;

	OpenCLDevice(const OpenCLDevice&) = delete;
	OpenCLDevice(OpenCLDevice&&) noexcept = default;

	OpenCLDevice& operator=(const OpenCLDevice&) = delete;
	OpenCLDevice& operator=(OpenCLDevice&&) noexcept = default;


};
