#pragma once

class SoulApplication {

public:

	SoulApplication();
	virtual ~SoulApplication() = default;

	virtual void Initialize();

	void Run();

protected:

	bool setupWindow;

private:

};
