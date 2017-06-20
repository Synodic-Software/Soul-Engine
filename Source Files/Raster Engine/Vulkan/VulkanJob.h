#pragma once

#include <string>

#include "Raster Engine\RasterJob.h"

class VulkanJob:public RasterJob {
public:
	VulkanJob();
	~VulkanJob();

	void AttachShaders(const std::vector<Shader*>&);
	void RegisterUniform(const std::string);
	void UploadGeometry(float*, uint, uint*, uint);
	void SetUniform(const std::string, RasterVariant);

	void Draw();
private:

};