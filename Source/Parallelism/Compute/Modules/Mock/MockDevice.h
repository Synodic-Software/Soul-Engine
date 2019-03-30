#pragma once


class MockDevice {

public:

    MockDevice() = default;
	virtual ~MockDevice() = default;

	MockDevice(const MockDevice&) = delete;
	MockDevice(MockDevice&&) noexcept = default;

	MockDevice& operator=(const MockDevice&) = delete;
	MockDevice& operator=(MockDevice&&) noexcept = default;


};