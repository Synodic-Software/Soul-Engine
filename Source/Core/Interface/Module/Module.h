#pragma once

#include "Core/Utility/Template/CRTP.h"


//Common interface for modules.
template<typename T>
class Module : public CRTP<T, Module> {

public:

	virtual ~Module() = default;

	Module(const Module&) = delete;
	Module(Module&&) noexcept = default;

	Module& operator=(const Module&) = delete;
	Module& operator=(Module&&) noexcept = default;


protected:

	Module() = default;


};
