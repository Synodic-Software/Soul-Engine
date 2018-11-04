#pragma once

enum class PlatformID { Windows, OSX, Linux, Unknown };


class Platform {

public:

	constexpr static PlatformID GetPlatform();
	constexpr static bool IsDesktop();
	constexpr static bool WithCLI();

private:

#ifdef __CYGWIN__

	constexpr static PlatformID platform = PlatformID::Linux;

#elif __linux__

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

#ifdef WITH_CLI

	constexpr static bool withCLI = true;

#else

	constexpr static bool withCLI = false;

#endif

};

constexpr PlatformID Platform::GetPlatform() {
	return platform;
}

constexpr bool Platform::IsDesktop() {
	return platform == PlatformID::Linux ||
		platform == PlatformID::Windows ||
		platform == PlatformID::OSX;
}

constexpr bool Platform::WithCLI() {
	return withCLI;
}
