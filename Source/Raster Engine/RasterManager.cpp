#include "RasterManager.h"


RasterManager::RasterManager() {
	
}

RasterManager::~RasterManager() {
	
}

void RasterManager::MakeContextCurrent() {


}

/*
 *    Creates a shader.
 *    @param	fileName	Filename of the file.
 *    @param	shaderT 	The shader t.
 *    @return	Null if it fails, else the new shader.
 */

Shader* RasterManager::CreateShader(const std::string& fileName, shader_t shaderT) {

	return nullptr;

}

/*
 *    Creates a buffer.
 *    @param	size	The size.
 *    @return	Null if it fails, else the new buffer.
 */

Buffer* RasterManager::CreateBuffer(uint size) {

	return nullptr;
}

/*
 *    Creates the job.
 *    @return	Null if it fails, else the new job.
 */

RasterJob* RasterManager::CreateJob() {

	return nullptr;

}

/* Sets window hints. */
void RasterManager::SetWindowHints() {
	raster.get()->SetWindowHints();
}

/*
 *    Resize window.
 *    @param [in,out]	window	If non-null, the window.
 *    @param 		 	x	  	The x coordinate.
 *    @param 		 	y	  	The y coordinate.
 */

void RasterManager::ResizeWindow(GLFWwindow* window, int x, int y) {
	raster.get()->ResizeWindow(window, x, y);
}

/*
 *    Builds a window.
 *    @param [in,out]	window	If non-null, the window.
 */

void RasterManager::BuildWindow(GLFWwindow* window) {
	raster.get()->BuildWindow(window);
}


/*
 *    Draws.
 *    @param [in,out]	window	If non-null, the window.
 *    @param [in,out]	job   	If non-null, the job.
 */

void RasterManager::Draw(GLFWwindow* window, RasterJob* job) {
	raster.get()->Draw(window, job);
}

/*
 *    Pre raster.
 *    @param [in,out]	window	If non-null, the window.
 */

void RasterManager::PreRaster(GLFWwindow* window) {
	raster.get()->PreRaster(window);
}

/*
 *    Posts a raster.
 *    @param [in,out]	window	If non-null, the window.
 */

void RasterManager::PostRaster(GLFWwindow* window) {
	raster.get()->PostRaster(window);
}

