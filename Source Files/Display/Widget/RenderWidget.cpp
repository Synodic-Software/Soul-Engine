#include "RenderWidget.h"
#include "Raster Engine\RasterBackend.h"

RenderWidget::RenderWidget()
{
	//attach shaders to render a quad and apply a texture
	widgetJob->AttachShaders({
		RasterBackend::CreateShader(window,"vertex-shader[Renderer].txt",VERTEX_SHADER),
		RasterBackend::CreateShader(window,"fragment-shader[Renderer].txt",FRAGMENT_SHADER)
	});

	//init all uniform data
	//(*widgetJob)[std::string("camera")]=
		//	cameraUniform = CUDAtoScreen->uniform("camera");
		//	modelUniform = CUDAtoScreen->uniform("model");
		//	screenUniform = CUDAtoScreen->uniform("screen");
		//	screenModUniform = CUDAtoScreen->uniform("screenMod");
}

RenderWidget::~RenderWidget()
{

}

void RenderWidget::Draw() {
	//	CUDAtoScreen->use();
	//	glBindVertexArray(vao);
	//	CUDAtoScreen->setUniform(cameraUniform, glm::ortho(0.0f, 1.0f, 0.0f, 1.0f, 2.0f, -2.0f));
	//	CUDAtoScreen->setUniform(modelUniform, glm::mat4());
	//	CUDAtoScreen->setUniform(screenUniform, originalScreen.x, originalScreen.y);
	//	CUDAtoScreen->setUniform(screenModUniform, modifiedScreen.x , modifiedScreen.y);
	//	//RenderJob->SwapResults(0,1);
	//	glDrawElements(GL_TRIANGLES, (6), GL_UNSIGNED_INT, (GLvoid*)0);
	//	glBindVertexArray(0);
	//	CUDAtoScreen->stopUsing();
}