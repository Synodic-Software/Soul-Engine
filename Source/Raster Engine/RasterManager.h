#pragma once

#include "Display\Window\Window.h"
#include "Raster Engine\Shader.h"
#include "Raster Engine\RasterJob.h"
#include "Raster Engine\Buffer.h"

#include "RasterBase.h"

/* Values that represent backend names. */
enum RasterAPI { Vulkan };

/* . */
class RasterManager {


public:

	static RasterManager& Instance() {
		static RasterManager instance;
		return instance;
	}

	RasterManager(RasterManager const&) = delete;
	void operator=(RasterManager const&) = delete;


	void		SetWindowHints();
	Shader*		CreateShader(const std::string&, shader_t);
	Buffer*		CreateBuffer(uint size);
	RasterJob*	CreateJob();
	void		MakeContextCurrent();
	void		ResizeWindow(GLFWwindow*, int, int);
	void		BuildWindow(GLFWwindow*);
	void		Draw(GLFWwindow*, RasterJob*);
	void		PreRaster(GLFWwindow*);
	void		PostRaster(GLFWwindow*);

private:

	RasterManager();
	~RasterManager();

	RasterAPI backend;
	std::unique_ptr<RasterBase> raster;

};
