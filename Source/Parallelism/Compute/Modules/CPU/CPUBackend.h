#pragma once

#include "Parallelism/Compute/ComputeBackend.h"


class CPUBackend : public ComputeBackend {

public:

	CPUBackend() = default;
	virtual ~CPUBackend() = default;

	CPUBackend(const CPUBackend&) = delete;
	CPUBackend(CPUBackend&&) noexcept = default;

	CPUBackend& operator=(const CPUBackend&) = delete;
	CPUBackend& operator=(CPUBackend&&) noexcept = default;


};
