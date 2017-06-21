#pragma once

#include <chrono>

/* A timer. */
/* A timer. */
class Timer {
public:
	/* Default constructor. */
	/* Default constructor. */
	Timer();
	/* Destructor. */
	/* Destructor. */
	~Timer();

	/* Resets this Timer. */
	/* Resets this Timer. */
	void Reset();

	/*
	 *    returns the elapsed time since the last reset in milliseconds.
	 *
	 *    @return	A double.
	 */

	double Elapsed();

private:
	/* Defines an alias representing the clock. */
	/* Defines an alias representing the clock. */
	typedef std::chrono::steady_clock Clock;
	/* Defines an alias representing the timestamp. */
	/* Defines an alias representing the timestamp. */
	typedef std::chrono::time_point<Clock> Timestamp;

	/* The tick hold */
	/* The tick hold */
	Timestamp tickHold;
};