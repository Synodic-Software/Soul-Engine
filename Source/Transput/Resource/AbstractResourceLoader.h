#pragma once

#include "Core/Interface/Project/Project.h"

#include "Core/Utility/CRTP/CRTP.h"

#include <iostream>
#include <fstream>
#include <filesystem>
#include <cassert>

template <typename T>
class AbstractResourceLoader : CRTP<T, AbstractResourceLoader> {

public:

	AbstractResourceLoader();

	virtual ~AbstractResourceLoader() = default;

	virtual void Load(const std::string_view&) = 0;

protected:

	static std::string_view extensions_[];

};

template <typename T>
AbstractResourceLoader<T>::AbstractResourceLoader()
{
}
