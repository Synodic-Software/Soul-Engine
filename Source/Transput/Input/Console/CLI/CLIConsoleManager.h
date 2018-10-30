#pragma once

#include "Composition/Event/EventManager.h"
#include "Transput/Input/Console/ConsoleManager.h"
#include "Core/Soul.h"

#include <iostream>

class CLIConsoleManager : public ConsoleManager {

public:

	CLIConsoleManager(EventManager*, Soul&);
	~CLIConsoleManager() override = default;

	bool Poll() override;

private:

	std::istream& istr_;
	std::ostream& ostr_;
	std::ostream& estr_;

};
