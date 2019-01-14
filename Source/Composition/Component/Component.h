#pragma once

#include "Core/Utility/Template/CRTP.h"


//The component class should hold no state TODO: because?
template<typename T>
class Component : CRTP<T, Component>
{

public:

	Component() = default;
	~Component() override = default;

};