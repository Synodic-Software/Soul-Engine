#include "ConsoleManager.h"

ConsoleManager::ConsoleManager(EventManager& eventManager, Soul& soul_ref) :
	eventManager_(eventManager),
	soul(soul_ref)
{
}
