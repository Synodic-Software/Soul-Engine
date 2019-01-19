#pragma once

#include "Core/Utility/Template/CRTP.h"

#include <memory>

class EntityManager;

template<typename T>
class Component : CRTP<T, Component>
{

public:

	Component() = default;
	~Component() override = default;

	Component(const Component &) = delete;
	Component(Component &&) noexcept = default;

	Component& operator=(const Component &) = delete;
	Component& operator=(Component &&) noexcept = default;


};