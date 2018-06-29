#pragma once

enum class DevelopPlatform { Windows, OSX, Linux, Unknown };
enum class ReleasePlatform { Windows, OSX, Linux };


class System {

public:

	static ReleasePlatform GetTarget();
	static bool IsDesktop();

private:

	static DevelopPlatform developPlatform;
	static ReleasePlatform deployPlatform;

};