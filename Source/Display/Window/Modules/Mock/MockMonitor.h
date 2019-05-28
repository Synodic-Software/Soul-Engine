#pragma once

#include "Display/Window/Monitor.h"
#include <string>

class MockMonitor : public Monitor
{

public:

	MockMonitor();
	virtual ~MockMonitor() = default;

	void Scale(float& xscale, float& yscale) const;
	
	void Position(int& xpos, int& ypos) const;

	void Size(int& width, int& height) const;
	void ColorBits(int& red, int& blue, int& green) const;
	void RefreshRate(int& refreshRate) const;

	std::string Name() const;

};
