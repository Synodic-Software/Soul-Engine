#pragma once

#include "Core/Soul.h"

class EventManager;
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
