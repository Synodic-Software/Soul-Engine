#pragma once

class Surface {

public:

	Surface() = default;
	virtual ~Surface() = default;

	Surface(const Surface&) = delete;
	Surface(Surface&& o) noexcept = default;

	Surface& operator=(const Surface&) = delete;
	Surface& operator=(Surface&& other) noexcept = default;

};
