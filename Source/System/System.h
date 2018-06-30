#pragma once

enum class Platform { Windows, OSX, Linux, Unknown };


class System {

public:

	constexpr static Platform GetPlatform() {
		return platform;
	}

	constexpr static bool IsDesktop() {
		return platform == Platform::Linux ||
			platform == Platform::Windows ||
			platform == Platform::OSX;
	}

private:

#ifdef __CYGWIN__

	constexpr static Platform platform = Platform::Linux;

#elif __GNU__

	constexpr static Platform platform = Platform::Linux;

#elif  _WIN64

	constexpr static Platform platform = Platform::Windows;

#elif __MACH__

	constexpr static Platform platform = Platform::OSX;

#elif __APPLE__

	constexpr static Platform platform = Platform::OSX;

#else

	constexpr static Platform platform = Platform::Unknown;

#endif

};