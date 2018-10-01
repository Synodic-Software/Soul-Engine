#pragma once

#include "glm/glm.hpp"
#include "Composition/Entity/Entity.h"

class SwapChain {

public:

	SwapChain() = default;
	virtual ~SwapChain() = default;

	SwapChain(const SwapChain&) = delete;
	SwapChain(SwapChain&& o) noexcept = default;

	SwapChain& operator=(const SwapChain&) = delete;
	SwapChain& operator=(SwapChain&& other) noexcept = default;

	virtual void Resize(Entity, glm::uvec2) = 0;
	virtual void Draw() = 0;

};
