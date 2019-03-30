#pragma once

#include "Parallelism/Compute/ComputeBuffer.h"


template <typename T>
class CPUBuffer : public ComputeBuffer<T> {

public:

	CPUBuffer() = default;
	virtual ~CPUBuffer() = default;

	CPUBuffer(const CPUBuffer&) = delete;
	CPUBuffer(CPUBuffer&&) noexcept = default;

	CPUBuffer& operator=(const CPUBuffer&) = delete;
	CPUBuffer& operator=(CPUBuffer&&) noexcept = default;


};
