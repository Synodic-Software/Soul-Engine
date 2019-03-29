#pragma once

#include "Core/Interface/Module/Module.h"
#include "Core/Utility/Template/CRTP.h"


template <typename T>
class ComputeBackend : public Module, public CRTP<T, ComputeBackend> {

public:

	ComputeBackend() = default;
	virtual ~ComputeBackend() = default;

	ComputeBackend(const ComputeBackend&) = delete;
	ComputeBackend(ComputeBackend&&) noexcept = default;

	ComputeBackend& operator=(const ComputeBackend&) = delete;
	ComputeBackend& operator=(ComputeBackend&&) noexcept = default;


};
