#pragma once

//Common interface for modules. aka plugins
class Module {

public:

	Module() = default;
	virtual ~Module() = default;

	Module(const Module&) = delete;
	Module(Module&&) noexcept = default;

	Module& operator=(const Module&) = delete;
	Module& operator=(Module&&) noexcept = default;


};
