#include "BDPTracer.h"

BDPTracer::BDPTracer( glm::vec2 screenSize1){
	focalDistance = 100;
	apertureRadius = 4 * MILLIMETER;
	samples = 1;
	rayDepth = 2;
	continuous = false;
	iterationCounter = 0;
	traceCounter = 0;
	startSwitch = 0;
	screenSize = screenSize1;
	/*shader = LoadShaders("N:\\Documents\\Dropbox\\Phasma\\Phasma-1.0\\Phasma-1.0\\Resource Files\\vertex-shader[BDPTracer].txt", 
		"N:\\Documents\\Dropbox\\Phasma\\Phasma-1.0\\Phasma-1.0\\Resource Files\\fragment-shader[BDPTracer].txt");
	texUniform = shader->uniform("tex");
	cameraUniform = shader->uniform("camera");
	modelUniform = shader->uniform("model");

	shaderCompute = LoadShaders("N:\\Documents\\Dropbox\\Phasma\\Phasma-1.0\\Phasma-1.0\\Resource Files\\compute-shader[BDPTracer].txt");
	nodeSizeUniform = shaderCompute->uniform("nodeSize");
	counterUniform1 = shaderCompute->uniform("counter");
	cameraApertureRadiusUniform = shaderCompute->uniform("cameraApertureRadius");
	cameraFocalDistanceUniform = shaderCompute->uniform("cameraFocalDistance");
	cameraPositionUniform = shaderCompute->uniform("cameraPosition");
	screenUniform = shaderCompute->uniform("screen");
	samplesUniform1 = shaderCompute->uniform("samples");
	horizontalAxisUniform = shaderCompute->uniform("horizontalAxis");
	verticalAxisUniform = shaderCompute->uniform("verticalAxis");
	middleUniform = shaderCompute->uniform("middle");
	horizontalUniform = shaderCompute->uniform("horizontal");
	verticalUniform = shaderCompute->uniform("vertical");


	shaderComputeColor = LoadShaders("N:\\Documents\\Dropbox\\Phasma\\Phasma-1.0\\Phasma-1.0\\Resource Files\\compute-shader[PTracerColor].txt");
	continuousUniform = shaderComputeColor->uniform("continuous");
	samplesUniform3 = shaderComputeColor->uniform("samples");
	counterUniform3 = shaderComputeColor->uniform("counter");

	display = createTex();

	Vertex fill = { { 0.0f, 0.0f, 0.0f, 1.0f }, { 0.0f, 1.0f }, { 0.0f, 0.0f, 0.0f } };
	BlockData[0] = fill;
	Vertex fill1 = { { 512.0f, 512.0f, 0.0f, 1.0f }, { 1.0f, 0.0f }, { 0.0f, 0.0f, 0.0f } };
	BlockData[1] = fill1;
	Vertex fill2 = { { 0.0f, 512.0f, 0.0f, 1.0f }, { 0.0f, 0.0f }, { 0.0f, 0.0f, 0.0f } };
	BlockData[2] = fill2;
	Vertex fill3 = { { 512.0f, 0.0f, 0.0f, 1.0f }, { 1.0f, 1.0f }, { 0.0f, 0.0f, 0.0f } };
	BlockData[3] = fill3;

	Indices[0] = 0;
	Indices[1] = 2;
	Indices[2] = 1;
	Indices[3] = 0;
	Indices[4] = 1;
	Indices[5] = 3;

	const size_t BufferSize = sizeof(BlockData);
	const size_t VertexSize = sizeof(BlockData[0]);

	glGenVertexArrays(1, &vao);
	glBindVertexArray(vao);

	glGenBuffers(1, &vbo);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glBufferData(GL_ARRAY_BUFFER, BufferSize, BlockData, GL_STATIC_DRAW);

	glEnableVertexAttribArray(shader->attrib("vert_VS_in"));
	glVertexAttribPointer(shader->attrib("vert_VS_in"), 4, GL_FLOAT, GL_FALSE, 9 * sizeof(GLfloat), NULL);
	glEnableVertexAttribArray(shader->attrib("texCoord_VS_in"));
	glVertexAttribPointer(shader->attrib("texCoord_VS_in"), 2, GL_FLOAT, GL_FALSE, 9 * sizeof(GLfloat), (const GLvoid*)(4 * sizeof(GLfloat)));

	glGenBuffers(1, &ibo);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(Indices), Indices, GL_STATIC_DRAW);

	glBindVertexArray(0);

	glGenBuffers(1, &propertiesSSBO);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, propertiesSSBO);
	glBufferData(GL_SHADER_STORAGE_BUFFER,
		screenSize.x*screenSize.y*samples*sizeof(asProp),
		NULL, GL_STATIC_DRAW);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

	glGenBuffers(1, &raysOriSSBO);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, raysOriSSBO);
	glBufferData(GL_SHADER_STORAGE_BUFFER,
		screenSize.x*screenSize.y*samples*sizeof(Origin),
		NULL, GL_STATIC_DRAW);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

	glGenBuffers(1, &raysDirSSBO);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, raysDirSSBO);
	glBufferData(GL_SHADER_STORAGE_BUFFER,
		screenSize.x*screenSize.y*samples*sizeof(Direction),
		NULL, GL_STATIC_DRAW);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

	glGenBuffers(1, &notAbsorpedSSBO);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, notAbsorpedSSBO);
	glBufferData(GL_SHADER_STORAGE_BUFFER,
		screenSize.x*screenSize.y*samples*sizeof(Direction),
		NULL, GL_STATIC_DRAW);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

	glGenBuffers(1, &colorSSBO);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, colorSSBO);
	glBufferData(GL_SHADER_STORAGE_BUFFER,
		screenSize.x*screenSize.y*sizeof(Direction),
		NULL, GL_STATIC_DRAW);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);*/

}
void BDPTracer::Update(double dt){

}

void BDPTracer::Render(Camera& camera, BVH* bvh){

	//glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, colorSSBO);
	//glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, notAbsorpedSSBO);
	//glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, raysDirSSBO);
	//glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, raysOriSSBO);
	//glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, propertiesSSBO);
	//glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, bvh->bvhVerticesSSBO);
	//glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, bvh->bvhIndicesSSBO);
	//glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 7, bvh->bvhNormalsSSBO);
	//glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 8, bvh->bvhTextureCoordsSSBO);
	//glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 9, bvh->BVHStructure);
	//glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 10, bvh->BVHAtomics);
	//glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 12, bvh->materialsSSBO);
	//glBindBufferBase(GL_UNIFORM_BUFFER, 11, bvh->texUBO);



	//// compute important values
	//glm::vec3 horizontalAxis = glm::cross(glm::normalize(camera.forward()), camera.up());
	//horizontalAxis = glm::normalize(horizontalAxis); // Important!
	//glm::vec3  verticalAxis = glm::cross(horizontalAxis, glm::normalize(camera.forward()));
	//verticalAxis = glm::normalize(verticalAxis); // verticalAxis is normalized by default, but normalize it explicitly just for good measure.

	//// compute point on image plane
	//glm::vec3  middle = camera.position() + glm::normalize(camera.forward());
	//glm::vec3  horizontal = horizontalAxis * glm::vec3(tan(camera.fieldOfView().x * 0.5 * (PI / 180.0)));
	//glm::vec3  vertical = verticalAxis * glm::vec3(tan(-(camera.fieldOfView().x / camera.viewportAspectRatio()) * 0.5 * (PI / 180.0)));

	//GLuint index = glGetUniformBlockIndex(shaderCompute->object(), "Samplers");
	//glUniformBlockBinding(shaderCompute->object(), index, 11);

	//shaderCompute->use();
	//shaderCompute->setUniform(counterUniform1, iterationCounter);
	//shaderCompute->setUniform(cameraApertureRadiusUniform, apertureRadius);
	//shaderCompute->setUniform(cameraFocalDistanceUniform, focalDistance);
	//shaderCompute->setUniform(cameraPositionUniform, camera.position().x, camera.position().y, camera.position().z);
	//shaderCompute->setUniform(screenUniform, (unsigned int)screenSize.x, (unsigned int)screenSize.y);
	//shaderCompute->setUniform(samplesUniform1, samples);
	//shaderCompute->setUniform(horizontalAxisUniform, horizontalAxis.x, horizontalAxis.y, horizontalAxis.z);
	//shaderCompute->setUniform(verticalAxisUniform, verticalAxis.x, verticalAxis.y, verticalAxis.z);
	//shaderCompute->setUniform(middleUniform, middle.x, middle.y, middle.z);
	//shaderCompute->setUniform(horizontalUniform, horizontal.x, horizontal.y, horizontal.z);
	//shaderCompute->setUniform(verticalUniform, vertical.x, vertical.y, vertical.z);
	//shaderCompute->setUniform(nodeSizeUniform, bvh->bufferSizeIn);

	//iterationCounter += 2;
	//glDispatchCompute(screenSize.x, screenSize.y, 1);
	//glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
	//shaderComputeI->stopUsing();

	//traceCounter++;



	//shaderComputeColor->use();



	//shaderComputeColor->setUniform(continuousUniform, continuous);
	//shaderComputeColor->setUniform(samplesUniform3, samples);
	//shaderComputeColor->setUniform(counterUniform3, traceCounter);
	//glBindImageTexture(5, display, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA32F);
	//glDispatchCompute(screenSize.x, screenSize.y, 1);
	//glMemoryBarrier(GL_ALL_BARRIER_BITS);
	//shaderComputeColor->stopUsing();

}
void BDPTracer::Draw(){
	/*shader->use();
	glBindVertexArray(vao);
	shader->setUniform(texUniform, 0);
	shader->setUniform(cameraUniform, glm::ortho(0.0f, 512.0f, 0.0f, 512.0f, -2.0f, 2.0f));
	shader->setUniform(modelUniform, position);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, display);
	glDrawElements(GL_TRIANGLES, (6), GL_UNSIGNED_INT, (GLvoid*)0);
	glBindVertexArray(0);
	shader->stopUsing();*/
}
void  BDPTracer::SwitchRender(){
	/*if (glfwGetTime()>startSwitch){
		startSwitch = glfwGetTime() + 1;
		continuous = !continuous;
		traceCounter = 0;
	}*/
}
GLuint BDPTracer::createTex(){
	GLuint texName;
	//glGenTextures(1, &texName);
	//glBindTexture(GL_TEXTURE_2D, texName);
	//glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	//glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	//glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	//glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	//glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, screenSize.x, screenSize.y, 0, GL_RGBA, GL_FLOAT, NULL);
	return texName;
}