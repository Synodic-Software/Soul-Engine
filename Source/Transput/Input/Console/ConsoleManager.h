#pragma once

#include "Composition/Event/EventManager.h"
#include "Transput/Input/Console/Parser/CommandParser.h"
#include "Soul.h"

class Soul;

class ConsoleManager {

public:

	ConsoleManager(EventManager&, Soul&);
	virtual ~ConsoleManager() = default;

	virtual void Poll() = 0;

protected:

	EventManager& eventManager_;
	Soul& soul;

};
