#pragma once
#include <string>

enum keyState { PRESS, REPEAT, RELEASE, OPEN };

class Key {

public:
	Key() : timeToRepeat(50.0f), state(OPEN), pressStart(0.0f) {
	}

	float timeToRepeat; //in milliseconds
	float pressStart;
	keyState state;

private:

};
