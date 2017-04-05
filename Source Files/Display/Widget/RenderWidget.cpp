#include "RenderWidget.h"
#include "Raster Engine\Buffer.h"
#include "GPGPU\GPUManager.h"
#include <iostream>

RenderWidget::RenderWidget()
{
	//attach shaders to render a quad and apply a texture
	widgetJob->AttachShaders({
		RasterBackend::CreateShader("../Resources/Shaders/vertex-shader[Renderer].glsl",VERTEX_SHADER),
		RasterBackend::CreateShader("../Resources/Shaders/fragment-shader[Renderer].glsl",FRAGMENT_SHADER)
	});


	//init all uniform data
	widgetJob->RegisterUniform(std::string("screen"));

	uint indices[6];
	float vertices[6 * 4];

	vertices[0] = -1.0f;
	vertices[1] = -1.0f;
	vertices[2] = 0.0f;
	vertices[3] = 1.0f;


	vertices[4] = 1.0f;
	vertices[5] = 1.0f;
	vertices[6] = 0.0f;
	vertices[7] = 1.0f;


	vertices[8] = -1.0f;
	vertices[9] = 1.0f;
	vertices[10] = 0.0f;
	vertices[11] = 1.0f;


	vertices[12] = 1.0f;
	vertices[13] = -1.0f;
	vertices[14] = 0.0f;
	vertices[15] = 1.0f;

	indices[0] = 0;
	indices[1] = 2;
	indices[2] = 1;
	indices[3] = 0;
	indices[4] = 1;
	indices[5] = 3;

	widgetJob->UploadGeometry(vertices, sizeof(vertices), indices, sizeof(indices));

}

RenderWidget::~RenderWidget()
{

}

void RenderWidget::Draw() {

	buffer->UnmapResources();
	buffer->BindData(0);
	widgetJob->Draw();

	GLenum x;
	if ((x = glGetError()) != GL_NO_ERROR) {
		std::cout << x << std::endl;
	}

	buffer->MapResources();

}

void RenderWidget::RecreateData() {

	buffer = GPUManager::CreateRasterBuffer(GPUManager::GetBestGPU(), size.x*size.y * sizeof(glm::vec4));

	if (currentSize != size) {
		currentSize = size;
		widgetJob->SetUniform(std::string("screen"), size);
	}

}
