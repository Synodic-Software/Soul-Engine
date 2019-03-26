#pragma once

#include "Rasterer/RasterDevice.h"

class MockDevice final: public RasterDevice {

public:

	MockDevice() = default;
	~MockDevice() override = default;

	MockDevice(const MockDevice &) = delete;
	MockDevice(MockDevice &&) noexcept = default;

	MockDevice& operator=(const MockDevice &) = delete;
	MockDevice& operator=(MockDevice &&) noexcept = default;

	void Synchronize() override;


};