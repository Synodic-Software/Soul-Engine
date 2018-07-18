#pragma once

#include "Core/Interface/Project/Project.h"

#include <filesystem>

class AbstractResourceLoader {
	
public:

	AbstractResourceLoader(const Project&);


protected:

	std::filesystem::path resourcePath_;
	std::filesystem::path engineResourcePath_;

	//TODO: allow multiple extensions registered per loader
	std::string_view extension_;

};
