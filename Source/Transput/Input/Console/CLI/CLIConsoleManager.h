#pragma once

#include "Composition/Event/EventManager.h"
#include "Transput/Input/Console/ConsoleManager.h"

#include <iostream>

class CLIConsoleManager : public ConsoleManager {

public:

	CLIConsoleManager(EventManager*);
	~CLIConsoleManager() override = default;

	bool Poll() override;

private:

	std::istream& istr_;
	std::ostream& ostr_;
	std::ostream& estr_;
};
