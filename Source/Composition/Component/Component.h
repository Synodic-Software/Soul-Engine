#pragma once

#include "Core/Utility/CRTP/CRTP.h"


//The component class should hold no per-instance state
template<typename T>
class Component : CRTP<T, Component>
{

public:

	Component() = default;

	virtual void Terminate() = 0;

};

template<typename T>
void Component<T>::Terminate() {
	this->Type().Terminate();
}
