#include "System.h"

#ifdef __CYGWIN__

DevelopPlatform System::developPlatform = DevelopPlatform::Linux;

#elif __GNU__

DevelopPlatform System::developPlatform = DevelopPlatform::Linux;

#elif  _WIN64

DevelopPlatform System::developPlatform = DevelopPlatform::Windows;

#elif __MACH__

DevelopPlatform System::developPlatform = DevelopPlatform::OSX;

#elif __APPLE__

DevelopPlatform System::developPlatform = DevelopPlatform::OSX;

#else

DevelopPlatform System::developPlatform = DevelopPlatform::Unknown;

#endif

ReleasePlatform System::deployPlatform;

ReleasePlatform System::GetTarget() {
	return deployPlatform;
}

bool System::IsDesktop() {
	return developPlatform == DevelopPlatform::Linux ||
		developPlatform == DevelopPlatform::Windows ||
		developPlatform == DevelopPlatform::OSX;
}