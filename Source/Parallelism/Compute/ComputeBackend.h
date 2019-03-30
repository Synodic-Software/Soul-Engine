#pragma once

#include "Core/Interface/Module/Module.h"
#include "ComputePolicy.h"

#include <memory>

class ComputeBackend : public Module<ComputeBackend> {

public:

	ComputeBackend() = default;
	virtual ~ComputeBackend() = default;

	ComputeBackend(const ComputeBackend&) = delete;
	ComputeBackend(ComputeBackend&&) noexcept = default;

	ComputeBackend& operator=(const ComputeBackend&) = delete;
	ComputeBackend& operator=(ComputeBackend&&) noexcept = default;


    template <typename KernelFunction, typename... Args>
	void Launch(const ComputePolicy& policy,
		const KernelFunction& kernel,
		Args&&... parameters) {
        
    }

    //Factory
	static std::shared_ptr<ComputeBackend> CreateModule();


};
