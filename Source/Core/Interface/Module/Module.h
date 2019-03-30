#pragma once

#include "Core/Utility/Template/CRTP.h"

#include <memory>
#include <vector>


//Common interface for modules.
template<typename T>
class Module : public CRTP<T, Module> {

public:

	Module() = default;
	virtual ~Module() = default;

	Module(const Module&) = delete;
	Module(Module&&) noexcept = default;

	Module& operator=(const Module&) = delete;
	Module& operator=(Module&&) noexcept = default;


protected:

    std::vector<std::unique_ptr<T>> modules;


};
