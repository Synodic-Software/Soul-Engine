#include "MockMonitor.h"

#include "Core/Utility/Exception/Exception.h"

MockMonitor::MockMonitor()
{
}

void MockMonitor::Scale(float&, float&) const
{

	throw NotImplemented();

}

void MockMonitor::Position(int&, int&) const
{

	throw NotImplemented();

}

void MockMonitor::Size(int&, int&) const
{

	throw NotImplemented();

}

void MockMonitor::ColorBits(int&, int&, int&) const
{

	throw NotImplemented();

}

void MockMonitor::RefreshRate(int&) const
{

	throw NotImplemented();

}

std::string MockMonitor::Name() const
{

	throw NotImplemented();

}
