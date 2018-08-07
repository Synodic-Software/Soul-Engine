#pragma once

#include <string_view>
#include <vulkan/vulkan.hpp>

class Pipeline {

public:

	//TODO abstract device
	Pipeline() = default;
	virtual ~Pipeline() = default;

	Pipeline(const Pipeline&) = delete;
	Pipeline(Pipeline&& o) noexcept = delete;

	Pipeline& operator=(const Pipeline&) = delete;
	Pipeline& operator=(Pipeline&& other) noexcept = delete;


};
