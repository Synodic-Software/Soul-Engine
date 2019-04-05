#pragma once

#include "Parallelism/Compute/ComputeDevice.h"
#include "Core/Utility/Exception/Exception.h"


class MockDevice : public ComputeDevice<MockDevice> {

public:

    MockDevice() = default;
	virtual ~MockDevice() = default;

	MockDevice(const MockDevice&) = delete;
	MockDevice(MockDevice&&) noexcept = default;

	MockDevice& operator=(const MockDevice&) = delete;
	MockDevice& operator=(MockDevice&&) noexcept = default;


    template <typename KernelFunction, typename... Args>
	void Launch(const ComputePolicy&, const KernelFunction&, Args&&...);


};


template <typename KernelFunction, typename... Args>
void MockDevice::Launch(const ComputePolicy& policy,
	const KernelFunction& kernel,
	Args&&... parameters) {

	throw NotImplemented();

}