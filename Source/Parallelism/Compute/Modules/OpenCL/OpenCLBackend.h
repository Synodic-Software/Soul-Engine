#pragma once

#include "Parallelism/Compute/ComputeBackend.h"


class OpenCLBackend : public ComputeBackend {

public:

	OpenCLBackend() = default;
	virtual ~OpenCLBackend() = default;

	OpenCLBackend(const OpenCLBackend&) = delete;
	OpenCLBackend(OpenCLBackend&&) noexcept = default;

	OpenCLBackend& operator=(const OpenCLBackend&) = delete;
	OpenCLBackend& operator=(OpenCLBackend&&) noexcept = default;


};
