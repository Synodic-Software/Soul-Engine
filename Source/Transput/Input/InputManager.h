#pragma once

#include "Composition/Event/EventManager.h"

class InputManager {

public:

	InputManager(EventManager&);
	virtual ~InputManager() = default;

	InputManager(const InputManager &) = delete;
	InputManager(InputManager&& o) noexcept = delete;

	InputManager& operator=(const InputManager&) = delete;
	InputManager& operator=(InputManager&& other) noexcept = delete;

	/*
	 * Window specific callbacks are registered within the function
	 *
	 * @param [in,out]	window	If non-null, the window.
	 */

	virtual void Poll() = 0;


protected:

	EventManager* eventManager_;

};