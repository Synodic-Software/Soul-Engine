#pragma once

#include "ComputePolicy.h"


class ComputeDevice {

public:

    ComputeDevice() = default;
	virtual ~ComputeDevice() = default;

	ComputeDevice(const ComputeDevice&) = delete;
	ComputeDevice(ComputeDevice&&) noexcept = default;

	ComputeDevice& operator=(const ComputeDevice&) = delete;
	ComputeDevice& operator=(ComputeDevice&&) noexcept = default;

    template <typename KernelFunction, typename... Args>
	void Launch(const ComputePolicy& policy,
		const KernelFunction& kernel,
		Args&&... parameters) {
	}


};
