#pragma once

#include "Composition/Event/EventManager.h"

class InputManager {

public:

	InputManager(EventManager&);
	virtual ~InputManager() = default;

	InputManager(const InputManager &) = delete;
	InputManager(InputManager&& o) noexcept = default;

	InputManager& operator=(const InputManager&) = delete;
	InputManager& operator=(InputManager&& other) noexcept = default;

	virtual bool Poll() = 0;


protected:

	EventManager* eventManager_;

};