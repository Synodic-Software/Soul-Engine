#pragma once

#include "Core/Interface/Module/Module.h"

#include <memory>


class ComputeBackend : public Module {

public:

	ComputeBackend() = default;
	virtual ~ComputeBackend() = default;

	ComputeBackend(const ComputeBackend&) = delete;
	ComputeBackend(ComputeBackend&&) noexcept = default;

	ComputeBackend& operator=(const ComputeBackend&) = delete;
	ComputeBackend& operator=(ComputeBackend&&) noexcept = default;

	//Factory
	static std::shared_ptr<ComputeBackend> CreateModule();


};
