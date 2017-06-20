//---------------------------------------------------------------------------------------------------
//@file	N:\Documents\Soul Engine\Source Files\Input\InputSet.h.
//Declares the input set class.

#pragma once

#include "Events\EventManager.h"
#include <string>

//An input set.
class InputSet
{

public:

	//---------------------------------------------------------------------------------------------------
	//Constructor.
	//@param	parameter1	The first parameter.

	InputSet(std::string);
	//Destructor.
	~InputSet();
	

private:
	//The name
	std::string name;
	//The key events
	EventManager::detail::EventMap keyEvents;

};