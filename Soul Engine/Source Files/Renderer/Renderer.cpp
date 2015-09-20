#include "Renderer.h"

Renderer::Renderer(){

	prevTime = 0.0f;
	changeCutoff = 0.1f;
	calcPass = 1;
	samplesMax = 4;
	samplesMin = 1;
	samples = samplesMin;
	CUDAtoScreen = LoadShaders("N:\\Documents\\Dropbox\\Phasma\\Phasma-1.0\\Phasma-1.0\\Resource Files\\vertex-shader[RayEngine].txt",
		"N:\\Documents\\Dropbox\\Phasma\\Phasma-1.0\\Phasma-1.0\\Resource Files\\fragment-shader[RayEngine].txt");
	cameraUniform = CUDAtoScreen->uniform("camera");
	modelUniform = CUDAtoScreen->uniform("model");
	screenUniform = CUDAtoScreen->uniform("screen");



	cuGraphicsGLRegisterImage(&graphicsResource
		, displayTexture
		, GL_TEXTURE_3D
		, CU_GRAPHICS_REGISTER_FLAGS_SURFACE_LDST);

	cuGraphicsMapResources(1, &graphicsResource, 0);

	cuGraphicsSubResourceGetMappedArray(&cudaDisplay
		, graphicsResource
		, 0, 0);

	cuGraphicsUnmapResources(1, &graphicsResource, 0);



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
	glEnableVertexAttribArray(CUDAtoScreen->attrib("texCoord_VS_in"));
	glVertexAttribPointer(CUDAtoScreen->attrib("texCoord_VS_in"), 2, GL_FLOAT, GL_FALSE, 6 * sizeof(GLfloat), (const GLvoid*)(4 * sizeof(GLfloat)));

	glGenBuffers(1, &ibo);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(Indices), Indices, GL_STATIC_DRAW);

	glBindVertexArray(0);
}

void Renderer::Render(glm::uvec2 screen, BVH* bvh, Camera& camera, double timeTarget){

		//float avgTime = 0;
		//for (std::list<float>::iterator itr = previousFrames.begin(); itr != previousFrames.end(); itr++){
		//	avgTime += *itr;
		//}
		//avgTime /= previousFrames.size();

		double newTime = glfwGetTime();
		/*if (calcPass==0){

		if (prevTime > timeTarget * (1.0f + changeCutoff)){
		if (samples > samplesMin){
		samples--;
		}
		}
		else if (prevTime < timeTarget){
		if (samples < samplesMax){
		samples++;
		}
		}
		}
		else{
		calcPass--;
		}*/

		glm::uvec2 modifiedScreen = glm::vec2(screen) / glm::vec2(1);
		GLuint renderBuffer = mainTracer->Draw(modifiedScreen, bvh, camera, samples);

		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, renderBuffer);

		CUDAtoScreen->use();
		glBindVertexArray(vao);
		CUDAtoScreen->setUniform(cameraUniform, glm::ortho(0.0f, 1.0f, 0.0f, 1.0f, -2.0f, 2.0f));
		CUDAtoScreen->setUniform(modelUniform, glm::mat4());
		CUDAtoScreen->setUniform(screenUniform, (float)modifiedScreen.x, (float)modifiedScreen.y);

		glDrawElements(GL_TRIANGLES, (6), GL_UNSIGNED_INT, (GLvoid*)0);
		glBindVertexArray(0);
		CUDAtoScreen->stopUsing();

		prevTime = glfwGetTime() - newTime;
}