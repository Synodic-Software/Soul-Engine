#include "Renderer.h"

Renderer::Renderer(Camera& camera, glm::uvec2 screen){




	modifiedScreen = screen;
	originalScreen = screen;
	camera.SetAspect((float)(screen.x) / (float)(screen.y));
	camera.resolution=screen;
	frameTime = 0.0f;
	changeCutoff = 0.1f;
	calcPass = 1;
	samplesMax = 4;
	samplesMin = 1;
	samples = samplesMin;
	CUDAtoScreen = LoadShaders("vertex-shader[Renderer].txt",
							   "fragment-shader[Renderer].txt");
	cameraUniform = CUDAtoScreen->uniform("camera");
	modelUniform = CUDAtoScreen->uniform("model");
	screenUniform = CUDAtoScreen->uniform("screen");
	screenModUniform = CUDAtoScreen->uniform("screenMod");

	glGenBuffers(1, &renderBuffer);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, renderBuffer);
	glBufferData(GL_SHADER_STORAGE_BUFFER,
		screen.x*screen.y*sizeof(float4),
		NULL, GL_STATIC_DRAW);

	glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

	CudaCheck(cudaGraphicsGLRegisterBuffer(&cudaBuffer
		, renderBuffer
		, cudaGraphicsRegisterFlagsWriteDiscard));


	CudaCheck(cudaGraphicsMapResources(1, &cudaBuffer, 0));
	size_t num_bytes;
	CudaCheck(cudaGraphicsResourceGetMappedPointer((void **)&bufferData, &num_bytes,
		cudaBuffer));

	RenderJob = RayEngine::AddRecurringRayJob(RayCOLOUR_TO_BUFFER, screen.x*screen.y, 1, &camera);

	RenderJob->resultsT = bufferData;

	Vertices[0] = 0.0f;
	Vertices[1] = 0.0f;
	Vertices[2] = 0.0f;
	Vertices[3] = 1.0f;

	Vertices[4] = 0.0f;
	Vertices[5] = 1.0f;



	Vertices[6] = 1.0f;
	Vertices[7] = 1.0f;
	Vertices[8] = 0.0f;
	Vertices[9] = 1.0f;

	Vertices[10] = 1.0f;
	Vertices[11] = 0.0f;



	Vertices[12] = 0.0f;
	Vertices[13] = 1.0f;
	Vertices[14] = 0.0f;
	Vertices[15] = 1.0f;

	Vertices[16] = 0.0f;
	Vertices[17] = 0.0f;



	Vertices[18] = 1.0f;
	Vertices[19] = 0.0f;
	Vertices[20] = 0.0f;
	Vertices[21] = 1.0f;

	Vertices[22] = 1.0f;
	Vertices[23] = 1.0f;


	Indices[0] = 0;
	Indices[1] = 2;
	Indices[2] = 1;
	Indices[3] = 0;
	Indices[4] = 1;
	Indices[5] = 3;

	const size_t BufferSize = sizeof(Vertices);
	const size_t VertexSize = sizeof(float) * 6;

	glGenVertexArrays(1, &vao);
	glBindVertexArray(vao);

	glGenBuffers(1, &vbo);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glBufferData(GL_ARRAY_BUFFER, BufferSize, Vertices, GL_STATIC_DRAW);

	glEnableVertexAttribArray(CUDAtoScreen->attrib("vert_VS_in"));
	glVertexAttribPointer(CUDAtoScreen->attrib("vert_VS_in"), 4, GL_FLOAT, GL_FALSE, 6 * sizeof(GLfloat), NULL);
	//glEnableVertexAttribArray(CUDAtoScreen->attrib("texCoord_VS_in"));
	//glVertexAttribPointer(CUDAtoScreen->attrib("texCoord_VS_in"), 2, GL_FLOAT, GL_FALSE, 6 * sizeof(GLfloat), (const GLvoid*)(4 * sizeof(GLfloat)));

	glGenBuffers(1, &ibo);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(Indices), Indices, GL_STATIC_DRAW);

	glBindVertexArray(0);
	newTime = glfwGetTime();
}

void Renderer::RenderRequestChange(glm::uvec2 screen, Camera& camera, double timeTarget){


	double oldTime = newTime;
	newTime = glfwGetTime();
	frameTime = newTime - oldTime;



	float aspectRatio = camera.GetAspect();
	uint newWidth = modifiedScreen.x;
	


	if (frameTime > timeTarget * (1.0f + changeCutoff)){
	if (samples > samplesMin){
	samples--;
	}
	else{
		newWidth = newWidth - (frameTime / timeTarget);
		if (newWidth<4){
			newWidth = 4;
		}
	}
	}
	else if (frameTime < timeTarget* (1.0f - changeCutoff)){
	if (samples < samplesMax){
	samples++;
	}
	else{
		newWidth = newWidth + (frameTime / timeTarget);
		if (newWidth>originalScreen.x){
			newWidth = originalScreen.x;
		}
	}
	}

	
	uint newHeight = glm::ceil(newWidth / aspectRatio);
	modifiedScreen = glm::uvec2(newWidth, newHeight);
	camera.resolution=modifiedScreen;
	RayEngine::ChangeJob(RenderJob, modifiedScreen.x*modifiedScreen.y,
		samples, &camera);

	
}


void Renderer::Render(){


		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, renderBuffer);

		CUDAtoScreen->use();
		glBindVertexArray(vao);
		CUDAtoScreen->setUniform(cameraUniform, glm::ortho(0.0f, 1.0f, 0.0f, 1.0f, 2.0f, -2.0f));
		CUDAtoScreen->setUniform(modelUniform, glm::mat4());
		CUDAtoScreen->setUniform(screenUniform, originalScreen.x, originalScreen.y);
		CUDAtoScreen->setUniform(screenModUniform, modifiedScreen.x , modifiedScreen.y);

		glDrawElements(GL_TRIANGLES, (6), GL_UNSIGNED_INT, (GLvoid*)0);
		glBindVertexArray(0);
		CUDAtoScreen->stopUsing();

		
}