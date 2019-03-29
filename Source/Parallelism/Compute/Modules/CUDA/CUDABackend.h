#pragma once

#include "Parallelism/Compute/ComputeBackend.h"


class CUDABackend : public ComputeBackend<CUDABackend> {

public:

	CUDABackend() = default;
	virtual ~CUDABackend() = default;

	CUDABackend(const CUDABackend&) = delete;
	CUDABackend(CUDABackend&&) noexcept = default;

	CUDABackend& operator=(const CUDABackend&) = delete;
	CUDABackend& operator=(CUDABackend&&) noexcept = default;


};
