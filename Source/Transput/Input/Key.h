#pragma once

#include "Core/Utility/Timer/Timer.h"

enum class KeyState { PRESS, REPEAT, RELEASE, OPEN };

class Key {

public:

	Key();

	float timeToRepeat; //in milliseconds
	Timer sincePress;

	KeyState state;

};
