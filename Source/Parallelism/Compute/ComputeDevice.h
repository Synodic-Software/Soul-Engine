#pragma once

#include <memory>


class ComputeDevice {

public:

    ComputeDevice() = default;
	virtual ~ComputeDevice() = default;

	ComputeDevice(const ComputeDevice&) = delete;
	ComputeDevice(ComputeDevice&&) noexcept = default;

	ComputeDevice& operator=(const ComputeDevice&) = delete;
	ComputeDevice& operator=(ComputeDevice&&) noexcept = default;


};