#pragma once

#include "Core/Utility/Template/CRTP.h"

class EntityManager;

template<typename T>
class Component : CRTP<T, Component>
{

public:

	Component() = default;
	~Component() override = default;

	Component(const Component &) = default;
	Component(Component &&) noexcept = default;

	Component& operator=(const Component &) = default;
	Component& operator=(Component &&) noexcept = default;


};