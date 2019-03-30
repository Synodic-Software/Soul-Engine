#pragma once

#include "Parallelism/Compute/ComputeBuffer.h"


template <typename T>
class CUDABuffer : public ComputeBuffer<T> {

public:

	CUDABuffer() = default;
	virtual ~CUDABuffer() = default;

	CUDABuffer(const CUDABuffer&) = delete;
	CUDABuffer(CUDABuffer&&) noexcept = default;

	CUDABuffer& operator=(const CUDABuffer&) = delete;
	CUDABuffer& operator=(CUDABuffer&&) noexcept = default;


};
