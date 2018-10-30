#include "CLIConsoleManager.h"

CLIConsoleManager::CLIConsoleManager(EventManager* eventManager, Soul& soul_ref) :
	ConsoleManager(eventManager, soul_ref),
	istr_(std::cin),
	ostr_(std::cout),
	estr_(std::cerr)
{
}

bool CLIConsoleManager::Poll() {

	return true;
}
