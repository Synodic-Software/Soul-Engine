#pragma once

#include "Utility\Includes\GLFWIncludes.h"
#include "Display\Window\Window.h"
#include "Raster Engine\RasterBackend.h"



class VulkanBackend : public RasterBackend::Backend {
public:
	VulkanBackend();
	~VulkanBackend();

	virtual void SetWindowHints(GLFWwindow*&);
	virtual void ResizeWindow(GLFWwindow*, int, int);
	virtual void BuildWindow(GLFWwindow*);
	void PreRaster(GLFWwindow*);
	void PostRaster(GLFWwindow*);
	virtual void Draw(GLFWwindow*, RasterJob*);

	template<typename Fn,
		typename ... Args>
		void RasterFunction(Fn && fn, Args && ... args) {


	}

private:

};