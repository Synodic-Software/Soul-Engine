#pragma once


class OpenCLDevice {

public:

    OpenCLDevice() = default;
	virtual ~OpenCLDevice() = default;

	OpenCLDevice(const OpenCLDevice&) = delete;
	OpenCLDevice(OpenCLDevice&&) noexcept = default;

	OpenCLDevice& operator=(const OpenCLDevice&) = delete;
	OpenCLDevice& operator=(OpenCLDevice&&) noexcept = default;


};