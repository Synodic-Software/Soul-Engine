#include "AbstractResourceLoader.h"

AbstractResourceLoader::AbstractResourceLoader(const Project& project):
	resourcePath_(project.GetDirectory() / "Resources"),
	engineResourcePath_(std::filesystem::current_path() / "Resources")
{	
}
