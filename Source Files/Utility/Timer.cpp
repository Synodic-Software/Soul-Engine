#include "Timer.h"

/* Default constructor. */
/* Default constructor. */
Timer::Timer() {
	Reset();
}
/* Destructor. */
/* Destructor. */
Timer::~Timer() {

}

/* Resets this Timer. */
/* Resets this Timer. */
void Timer::Reset() {

	tickHold = Clock::now();

}

/*
 *    Gets the elapsed.
 *
 *    @return	A double.
 */

double Timer::Elapsed() {

	std::chrono::duration<double, std::milli> ms = Clock::now() - tickHold;
	return ms.count();

}
