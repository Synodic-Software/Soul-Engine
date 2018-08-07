#pragma once

class RasterDevice {

public:

	RasterDevice() = default;
	virtual ~RasterDevice() = default;

	RasterDevice(const RasterDevice&) = delete;
	RasterDevice(RasterDevice&& o) noexcept = default;

	RasterDevice& operator=(const RasterDevice&) = delete;
	RasterDevice& operator=(RasterDevice&& other) noexcept = default;

};
