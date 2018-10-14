#pragma once

#include "Composition/Event/EventManager.h"

class ConsoleManager {

public:

	ConsoleManager(EventManager*);
	virtual ~ConsoleManager() = default;

	virtual bool Poll() = 0;

protected:

	EventManager* eventManager_;

};
