#include "Timer.h"

Timer::Timer() {
	Reset();
}
Timer::~Timer() {

}

void Timer::Reset() {

	tickHold = Clock::now();

}

double Timer::Elapsed() {

	std::chrono::duration<double, std::milli> ms = Clock::now() - tickHold;
	return ms.count();

}
