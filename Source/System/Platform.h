#pragma once

enum class PlatformID { Windows, OSX, Linux, Android, Unknown };

class Platform {

public:

	constexpr static PlatformID GetPlatform();
	constexpr static bool IsDesktop();
	constexpr static bool WithCLI();

private:

#ifdef __linux__ && !__ANDROID__

	constexpr static PlatformID platform = PlatformID::Linux;

#elif  _WIN64

	constexpr static PlatformID platform = PlatformID::Windows;

#elif __APPLE__

	constexpr static PlatformID platform = PlatformID::OSX;

#elif __ANDROID__ 

	constexpr static PlatformID platform = PlatformID::Android;

#else

	constexpr static PlatformID platform = PlatformID::Unknown;

#endif

//TODO: Remove macro. CLI should not be handled by platform, but by Application
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

//TODO: Does not belong here
constexpr bool Platform::WithCLI() {
	return withCLI;
}
