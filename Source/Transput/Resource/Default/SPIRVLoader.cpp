#include "SPIRVLoader.h"

SPIRVLoader::SPIRVLoader(const Project& project):
	AbstractResourceLoader(project)
{	
	extension_ = "spv";
}
