#pragma once

#include "Parallelism/Compute/ComputeDevice.h"


class CUDADevice : public ComputeDevice {

public:

    CUDADevice() = default;
	virtual ~CUDADevice() = default;

	CUDADevice(const CUDADevice&) = delete;
	CUDADevice(CUDADevice&&) noexcept = default;

	CUDADevice& operator=(const CUDADevice&) = delete;
	CUDADevice& operator=(CUDADevice&&) noexcept = default;


};