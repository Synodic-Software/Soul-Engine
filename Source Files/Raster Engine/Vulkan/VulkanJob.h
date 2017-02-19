#pragma once

#include <string>

#include "Raster Engine\RasterJob.h"

class VulkanJob:public RasterJob {
public:
	VulkanJob(const std::vector<Shader>&);
	~VulkanJob();
private:

};