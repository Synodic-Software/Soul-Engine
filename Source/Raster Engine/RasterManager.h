#pragma once

#include "RHI/RHI.h"

#include "glm/glm.hpp"

class RasterManager {


public:

	static RasterManager& Instance() {
		static RasterManager instance;
		return instance;
	}

	RasterManager(RasterManager const&) = delete;
	void operator=(RasterManager const&) = delete;

	//Draw and Update steps called from the main loop
	void PreRaster();
	void Raster();
	void PostRaster();

private:

	RasterManager();
	~RasterManager();

	RHI rasterAPI;

};
