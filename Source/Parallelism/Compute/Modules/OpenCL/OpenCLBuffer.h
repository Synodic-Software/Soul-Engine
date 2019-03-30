#pragma once

#include "Parallelism/Compute/ComputeBuffer.h"


template <typename T>
class OpenCLBuffer : public ComputeBuffer<T>{

public:

	OpenCLBuffer() = default;
	virtual ~OpenCLBuffer() = default;

	OpenCLBuffer(const OpenCLBuffer&) = delete;
	OpenCLBuffer(OpenCLBuffer&&) noexcept = default;

	OpenCLBuffer& operator=(const OpenCLBuffer&) = delete;
	OpenCLBuffer& operator=(OpenCLBuffer&&) noexcept = default;


};
