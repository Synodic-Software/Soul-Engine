#pragma once

#include "Parallelism/Compute/ComputeDevice.h"


class CPUDevice : public ComputeDevice {

public:

    CPUDevice() = default;
	virtual ~CPUDevice() = default;

	CPUDevice(const CPUDevice&) = delete;
	CPUDevice(CPUDevice&&) noexcept = default;

	CPUDevice& operator=(const CPUDevice&) = delete;
	CPUDevice& operator=(CPUDevice&&) noexcept = default;


};