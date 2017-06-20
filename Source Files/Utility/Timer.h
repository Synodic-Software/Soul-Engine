#pragma once

#include <chrono>

class Timer {
public:
	Timer();
	~Timer();

	void Reset();

	//returns the elapsed time since the last reset in milliseconds
	double Elapsed();

private:
	typedef std::chrono::steady_clock Clock;
	typedef std::chrono::time_point<Clock> Timestamp;

	Timestamp tickHold;
};