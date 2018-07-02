#pragma once

enum class PlatformID { Windows, OSX, Linux, Unknown };


class Platform {

public:

	constexpr static PlatformID GetPlatform() {
		return platform;
	}

	constexpr static bool IsDesktop() {
		return platform == PlatformID::Linux ||
			platform == PlatformID::Windows ||
			platform == PlatformID::OSX;
	}

private:

#ifdef __CYGWIN__

	constexpr static PlatformID platform = PlatformID::Linux;

#elif __GNU__

	constexpr static PlatformID platform = PlatformID::Linux;

#elif  _WIN64

	constexpr static PlatformID platform = PlatformID::Windows;

#elif __MACH__

	constexpr static PlatformID platform = PlatformID::OSX;

#elif __APPLE__

	constexpr static PlatformID platform = PlatformID::OSX;

#else

	constexpr static PlatformID platform = PlatformID::Unknown;

#endif

};