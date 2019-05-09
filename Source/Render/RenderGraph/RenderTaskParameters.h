#pragma once

#include <string>

class RenderTaskParameters {

public:

	RenderTaskParameters() = default;
	~RenderTaskParameters() = default;

	RenderTaskParameters(const RenderTaskParameters &) = delete;
	RenderTaskParameters(RenderTaskParameters &&) noexcept = default;

	RenderTaskParameters& operator=(const RenderTaskParameters &) = delete;
	RenderTaskParameters& operator=(RenderTaskParameters &&) noexcept = default;

	std::string name;

};
