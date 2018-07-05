#pragma once

#include "Composition/Event/EventManager.h"

class InputManager {

public:

	InputManager(EventManager&);
	virtual ~InputManager() = default;

	InputManager(InputManager const&) = delete;
	InputManager(InputManager&& o) = delete;

	InputManager& operator=(InputManager const&) = delete;
	InputManager& operator=(InputManager&& other) = delete;

	/*
	 * Window specific callbacks are registered within the function
	 *
	 * @param [in,out]	window	If non-null, the window.
	 */

	virtual void Poll() = 0;


protected:

	EventManager* eventManager_;

};