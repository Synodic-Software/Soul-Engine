#pragma once

#include <boost/dll/alias.hpp>

//Common interface for modules. aka plugins
class Module {

public:

	Module() = default;
	virtual ~Module() = 0;

	Module(const Module&) = delete;
	Module(Module&&) noexcept = delete;

	Module& operator=(const Module&) = delete;
	Module& operator=(Module&&) noexcept = delete;


};
