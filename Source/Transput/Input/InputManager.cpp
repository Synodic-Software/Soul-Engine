#include "InputManager.h"
#include "Composition/Event/EventManager.h"

InputManager::InputManager(EventManager& eventManager) :
	eventManager_(&eventManager)
{	
}


