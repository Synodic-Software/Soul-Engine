#pragma once

#include <string>

#include "Raster Engine\RasterJob.h"

class VulkanJob:public RasterJob {
public:
	VulkanJob();
	~VulkanJob();

	void AttachShaders(const std::vector<Shader*>&);
private:

};