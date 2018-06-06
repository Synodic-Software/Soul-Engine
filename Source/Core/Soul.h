#pragma once

#include "SoulParameters.h"
#include <memory>

class Soul {

public:

	Soul(SoulParameters&);
	~Soul();

	Soul(Soul&&) noexcept;
	Soul& operator=(Soul&&) noexcept;

	void Initialize() const;
	void Terminate() const;

	void Run();


private:

	SoulParameters& parameters;

	//hidden Soul services and modules
	class Implementation;
	std::unique_ptr<Implementation> detail;

};
