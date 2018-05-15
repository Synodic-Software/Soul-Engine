#pragma once

#include "Core/Utility/Timer.h"

#include <string>

enum keyState { PRESS, REPEAT, RELEASE, OPEN };

class Key {

public:
	Key() : timeToRepeat(50.0f), state(OPEN) {
	}

	float timeToRepeat; //in milliseconds
	Timer sincePress;

	keyState state;

private:

};
