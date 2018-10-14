#include "CLIConsoleManager.h"

CLIConsoleManager::CLIConsoleManager(EventManager* eventManager) :
	ConsoleManager(eventManager),
	istr_(std::cin),
	ostr_(std::cout),
	estr_(std::cerr)
{
}

bool CLIConsoleManager::Poll() {

	return true;
}
