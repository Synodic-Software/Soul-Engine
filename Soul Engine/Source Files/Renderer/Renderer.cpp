#include "Renderer.h"
#include "CUDA/Renderer.cuh"

Renderer::Renderer(Camera& camera, glm::uvec2 screen){
	iCounter = 1;
	debug = false;
	modifiedScreen = screen;
	originalScreen = screen;
	camera.SetAspect((float)(screen.x) / (float)(screen.y));
	camera.resolution=screen;
	frameTime = 0.0f;
	changeCutoff = 0.05f;
	targetFPS = 60.0f;
	samples = 4;
	//samples cannot be a float because finding a constant random number across different threads becomes too time consuming 

	CUDAtoScreen = LoadShaders("vertex-shader[Renderer].txt",
							   "fragment-shader[Renderer].txt");
	cameraUniform = CUDAtoScreen->uniform("camera");
	modelUniform = CUDAtoScreen->uniform("model");
	screenUniform = CUDAtoScreen->uniform("screen");
	screenModUniform = CUDAtoScreen->uniform("screenMod");

	glGenBuffers(1, &renderBufferA);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, renderBufferA);
	glBufferData(GL_SHADER_STORAGE_BUFFER,
		originalScreen.x*originalScreen.y*sizeof(glm::vec4),
		NULL, GL_STATIC_DRAW);

	glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

	CudaCheck(cudaGraphicsGLRegisterBuffer(&cudaBuffer
		, renderBufferA
		, cudaGraphicsRegisterFlagsWriteDiscard));


	CudaCheck(cudaGraphicsMapResources(1, &cudaBuffer, 0));
	size_t num_bytes;
	CudaCheck(cudaGraphicsResourceGetMappedPointer((void **)&bufferData, &num_bytes,
		cudaBuffer));

	CudaCheck(cudaGraphicsUnmapResources(1, &cudaBuffer, 0));
	RenderJob = RayEngine::AddRayJob(RayCOLOUR, screen.x*screen.y, samples, &camera,2);

	CudaCheck(cudaFree(RenderJob->GetResultPointer(0)));
	RenderJob->GetResultPointer(0) = bufferData;

	Vertices[0] = 0.0f;
	Vertices[1] = 0.0f;
	Vertices[2] = 0.0f;
	Vertices[3] = 1.0f;



	Vertices[4] = 1.0f;
	Vertices[5] = 1.0f;
	Vertices[6] = 0.0f;
	Vertices[7] = 1.0f;



	Vertices[8] = 0.0f;
	Vertices[9] = 1.0f;
	Vertices[10] = 0.0f;
	Vertices[11] = 1.0f;



	Vertices[12] = 1.0f;
	Vertices[13] = 0.0f;
	Vertices[14] = 0.0f;
	Vertices[15] = 1.0f;


	Indices[0] = 0;
	Indices[1] = 2;
	Indices[2] = 1;
	Indices[3] = 0;
	Indices[4] = 1;
	Indices[5] = 3;

	const size_t BufferSize = sizeof(Vertices);
	const size_t VertexSize = sizeof(GLfloat) * 4;

	glGenVertexArrays(1, &vao);
	glBindVertexArray(vao);

	glGenBuffers(1, &vbo);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glBufferData(GL_ARRAY_BUFFER, BufferSize, Vertices, GL_STATIC_DRAW);

	glEnableVertexAttribArray(CUDAtoScreen->attrib("vert_VS_in"));
	glVertexAttribPointer(CUDAtoScreen->attrib("vert_VS_in"), 4, GL_FLOAT, GL_FALSE, VertexSize, NULL);

	glGenBuffers(1, &ibo);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(Indices), Indices, GL_STATIC_DRAW);

	glBindVertexArray(0);
	newTime = glfwGetTime();
}

void Renderer::RenderSetup(const glm::uvec2& screen, Camera* camera, double timeTarget, float scroll){


	double oldTime = newTime;
	newTime = glfwGetTime();
	double smoothing = 0.85;

	double frameTop=newTime - oldTime;
	frameTime = (frameTime * smoothing) + (frameTop * (1.0 - smoothing));

	float aspectRatio = camera->GetAspect();
	uint newWidth = (uint)glm::ceil(originalScreen.x*scroll);
	
	uint workCalc = RenderJob->GetSampleAmount()*RenderJob->GetRayAmount();

	uint workTarget = (1000.0f / timeTarget) / frameTime;

	/*if (workTarget - (workTarget*changeCutoff) > workCalc){
		newWidth = newWidth*(1.0+changeCutoff);
	}
	else if (workTarget + (workTarget*changeCutoff) < workCalc){
		newWidth = newWidth/(1.0 + changeCutoff);
	}*/


	if (newWidth<32){
		newWidth = 32;
	}
	else if (newWidth>originalScreen.x){
		newWidth = originalScreen.x;
	}

	uint newHeight = (uint) glm::ceil(newWidth / aspectRatio);
	modifiedScreen = glm::uvec2(newWidth, newHeight);
	camera->resolution=modifiedScreen;
	RayEngine::ChangeJob(RenderJob, (modifiedScreen.x*modifiedScreen.y),
		samples, camera);
	//RenderJob->GetSampleAmount()=0.1f;
	CudaCheck(cudaGraphicsMapResources(1, &cudaBuffer, 0));
	size_t num_bytes;
	CudaCheck(cudaGraphicsResourceGetMappedPointer((void **)&bufferData, &num_bytes,
		cudaBuffer));

}

void Renderer::Render(bool integrate){

	if (integrate){
		Integrate(RenderJob, iCounter);
		iCounter++;
	}
	else{
		iCounter = 1;
	}		
	
	CudaCheck(cudaGraphicsUnmapResources(1, &cudaBuffer, 0));

	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, renderBufferA);

	CUDAtoScreen->use();
	glBindVertexArray(vao);
	CUDAtoScreen->setUniform(cameraUniform, glm::ortho(0.0f, 1.0f, 0.0f, 1.0f, 2.0f, -2.0f));
	CUDAtoScreen->setUniform(modelUniform, glm::mat4());
	CUDAtoScreen->setUniform(screenUniform, originalScreen.x, originalScreen.y);
	CUDAtoScreen->setUniform(screenModUniform, modifiedScreen.x , modifiedScreen.y);
	//RenderJob->SwapResults(0,1);
	glDrawElements(GL_TRIANGLES, (6), GL_UNSIGNED_INT, (GLvoid*)0);
	glBindVertexArray(0);
	CUDAtoScreen->stopUsing();

	//RenderJob->SwapResults(0, 1);

}