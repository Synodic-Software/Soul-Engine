//#include "PTracer.h"
//
//PTracer::PTracer(glm::vec2 screenSize1){
//	focalDistance=100;
//	apertureRadius=4* MILLIMETER;
//	samples=1;
//	rayDepth=4;
//	continuous = false;
//	iterationCounter=0;
//	traceCounter = 0; 
//	startSwitch = 0;
//	screenSize=screenSize1;
//	/*shader = LoadShaders("N:\\Documents\\Dropbox\\Phasma\\Phasma-1.0\\Phasma-1.0\\Resource Files\\vertex-shader[PTracer].txt",
//		"N:\\Documents\\Dropbox\\Phasma\\Phasma-1.0\\Phasma-1.0\\Resource Files\\fragment-shader[PTracer].txt");
//	texUniform = shader->uniform("tex");
//	cameraUniform = shader->uniform("camera");
//	modelUniform = shader->uniform("model");
//
//	shaderCompute = LoadShaders("N:\\Documents\\Dropbox\\Phasma\\Phasma-1.0\\Phasma-1.0\\Resource Files\\compute-shader[PTracer].txt");
//	samplesUniform2 = shaderCompute->uniform("samples");
//	nodeSizeUniform = shaderCompute->uniform("nodeSize");
//	counterUniform2 = shaderCompute->uniform("counter");
//
//	shaderComputeI = LoadShaders("N:\\Documents\\Dropbox\\Phasma\\Phasma-1.0\\Phasma-1.0\\Resource Files\\compute-shader[PTracerI].txt"); 
//	counterUniform1 = shaderComputeI->uniform("counter");
//	cameraApertureRadiusUniform = shaderComputeI->uniform("cameraApertureRadius");
//	cameraFocalDistanceUniform = shaderComputeI->uniform("cameraFocalDistance");
//	cameraPositionUniform = shaderComputeI->uniform("cameraPosition");
//	screenUniform = shaderComputeI->uniform("screen");
//	samplesUniform1 = shaderComputeI->uniform("samples");
//	horizontalAxisUniform = shaderComputeI->uniform("horizontalAxis");
//	verticalAxisUniform = shaderComputeI->uniform("verticalAxis");
//	middleUniform = shaderComputeI->uniform("middle");
//	horizontalUniform = shaderComputeI->uniform("horizontal");
//	verticalUniform = shaderComputeI->uniform("vertical");
//
//
//	shaderComputeColor = LoadShaders("N:\\Documents\\Dropbox\\Phasma\\Phasma-1.0\\Phasma-1.0\\Resource Files\\compute-shader[PTracerColor].txt");
//	continuousUniform = shaderComputeColor->uniform("continuous");
//	samplesUniform3 = shaderComputeColor->uniform("samples");
//	counterUniform3 = shaderComputeColor->uniform("counter");
//
//	display=createTex();
//
//	Vertex fill={{0.0f,0.0f,0.0f,1.0f},{0.0f,1.0f},{0.0f,0.0f,0.0f}};
//	BlockData[0]=fill;
//	Vertex fill1={{512.0f,512.0f,0.0f,1.0f},{1.0f,0.0f},{0.0f,0.0f,0.0f}};
//	BlockData[1]=fill1;
//	Vertex fill2={{0.0f,512.0f,0.0f,1.0f},{0.0f,0.0f},{0.0f,0.0f,0.0f}};
//	BlockData[2]=fill2;
//	Vertex fill3={{512.0f,0.0f,0.0f,1.0f},{1.0f,1.0f},{0.0f,0.0f,0.0f}};
//	BlockData[3]=fill3;
//	
//	Indices[0]=0;
//	Indices[1]=2;
//	Indices[2]=1;
//	Indices[3]=0;
//	Indices[4]=1;
//	Indices[5]=3;
//
//	const size_t BufferSize = sizeof(BlockData);
//	const size_t VertexSize = sizeof(BlockData[0]);	
//
//	glGenVertexArrays(1, &vao);
//	glBindVertexArray(vao);
//	
//	glGenBuffers(1, &vbo);
//	glBindBuffer(GL_ARRAY_BUFFER, vbo);
//	glBufferData(GL_ARRAY_BUFFER, BufferSize, BlockData, GL_STATIC_DRAW);
//
//	glEnableVertexAttribArray(shader->attrib("vert_VS_in"));
//    glVertexAttribPointer(shader->attrib("vert_VS_in"), 4, GL_FLOAT, GL_FALSE, 9*sizeof(GLfloat), NULL);
//	glEnableVertexAttribArray(shader->attrib("texCoord_VS_in"));
//    glVertexAttribPointer(shader->attrib("texCoord_VS_in"), 2, GL_FLOAT, GL_FALSE, 9*sizeof(GLfloat),(const GLvoid*)(4 * sizeof(GLfloat)));
//
//	glGenBuffers(1, &ibo);
//	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);
//	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(Indices), Indices, GL_STATIC_DRAW);
//
//	glBindVertexArray(0);
//
//	glGenBuffers( 1, &propertiesSSBO);
//	glBindBuffer( GL_SHADER_STORAGE_BUFFER, propertiesSSBO );
//	glBufferData( GL_SHADER_STORAGE_BUFFER,
//	screenSize.x*screenSize.y*samples*sizeof(asProp),
//	NULL, GL_STATIC_DRAW );
//	glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
//
//	glGenBuffers( 1, &raysOriSSBO);
//	glBindBuffer( GL_SHADER_STORAGE_BUFFER, raysOriSSBO );
//	glBufferData( GL_SHADER_STORAGE_BUFFER,
//	screenSize.x*screenSize.y*samples*sizeof(Origin),
//	NULL, GL_STATIC_DRAW );
//	glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
//
//	glGenBuffers( 1, &raysDirSSBO);
//	glBindBuffer( GL_SHADER_STORAGE_BUFFER, raysDirSSBO );
//	glBufferData( GL_SHADER_STORAGE_BUFFER,
//	screenSize.x*screenSize.y*samples*sizeof(Direction),
//	NULL, GL_STATIC_DRAW );
//	glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
//
//	glGenBuffers( 1, &notAbsorpedSSBO);
//	glBindBuffer( GL_SHADER_STORAGE_BUFFER, notAbsorpedSSBO );
//	glBufferData( GL_SHADER_STORAGE_BUFFER,
//	screenSize.x*screenSize.y*samples*sizeof(Direction),
//	NULL, GL_STATIC_DRAW );
//	glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
//
//	glGenBuffers( 1, &colorSSBO);
//	glBindBuffer( GL_SHADER_STORAGE_BUFFER, colorSSBO );
//	glBufferData( GL_SHADER_STORAGE_BUFFER,
//	screenSize.x*screenSize.y*sizeof(Direction),
//	NULL, GL_STATIC_DRAW );
//	glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);*/
//
//}
//	void PTracer::Update(double dt){
//
//	}
//
//	void PTracer::Render(Camera& camera, BVH* bvh){
//
//	//glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, colorSSBO);
//	//glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, notAbsorpedSSBO);
//	//glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, raysDirSSBO);
//	//glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, raysOriSSBO);
//	//glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, propertiesSSBO);
//	//glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, bvh->bvhVerticesSSBO);
//	//glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, bvh->bvhIndicesSSBO);
//	//glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 7, bvh->bvhNormalsSSBO);
//	//glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 8, bvh->bvhTextureCoordsSSBO);
//	//glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 9, bvh->BVHStructure);
//	//glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 10, bvh->BVHAtomics);
//	//glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 12, bvh->materialsSSBO);
//	//glBindBufferBase(GL_UNIFORM_BUFFER, 11, bvh->texUBO);
//
//
//
//	//// compute important values
//	//glm::vec3 horizontalAxis = glm::cross(glm::normalize(camera.forward()), camera.up());
//	//horizontalAxis = glm::normalize(horizontalAxis);
//	//glm::vec3  verticalAxis = glm::cross(horizontalAxis, glm::normalize(camera.forward()));
//	//verticalAxis = glm::normalize(verticalAxis);
//
//	//// compute point on image plane
//	//glm::vec3  middle = camera.position() + glm::normalize(camera.forward());
//	//glm::vec3  horizontal = horizontalAxis * glm::vec3(tan(camera.fieldOfView().x * 0.5 * (PI / 180.0)));
//	//glm::vec3  vertical = verticalAxis * glm::vec3(tan(-(camera.fieldOfView().x / camera.viewportAspectRatio()) * 0.5 * (PI / 180.0)));
//
//	//shaderComputeI->use();
//	//shaderComputeI->setUniform(counterUniform1, iterationCounter);
//	//shaderComputeI->setUniform(cameraApertureRadiusUniform, apertureRadius);
//	//shaderComputeI->setUniform(cameraFocalDistanceUniform, focalDistance);
//	//shaderComputeI->setUniform(cameraPositionUniform, camera.position().x, camera.position().y, camera.position().z);
//	//shaderComputeI->setUniform(screenUniform, (unsigned int)screenSize.x, (unsigned int)screenSize.y);
//	//shaderComputeI->setUniform(samplesUniform1, samples);
//	//shaderComputeI->setUniform(horizontalAxisUniform, horizontalAxis.x, horizontalAxis.y, horizontalAxis.z);
//	//shaderComputeI->setUniform(verticalAxisUniform, verticalAxis.x, verticalAxis.y, verticalAxis.z);
//	//shaderComputeI->setUniform(middleUniform, middle.x, middle.y, middle.z);
//	//shaderComputeI->setUniform(horizontalUniform, horizontal.x, horizontal.y, horizontal.z);
//	//shaderComputeI->setUniform(verticalUniform, vertical.x, vertical.y, vertical.z);
//
//	//iterationCounter++;
//	//glDispatchCompute( screenSize.x, screenSize.y, 1 );
//	//glMemoryBarrier( GL_SHADER_STORAGE_BARRIER_BIT | GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
//	//shaderComputeI->stopUsing();
//
//	//
//
//	//shaderCompute->use();
//
//	//GLuint index = glGetUniformBlockIndex(shaderCompute->object(),"Samplers");
//	//glUniformBlockBinding(shaderCompute->object(),index,11);
//
//	//shaderCompute->setUniform(samplesUniform2, samples);
//	//shaderCompute->setUniform(nodeSizeUniform, bvh->bufferSizeIn);
//	//for (int rayDepthC = 0; rayDepthC < rayDepth; rayDepthC++) {
//	//iterationCounter++;
//	//shaderCompute->setUniform(counterUniform2, iterationCounter);
//	//glDispatchCompute( screenSize.x, screenSize.y, 1 );
//	//glMemoryBarrier( GL_ALL_BARRIER_BITS );
//
//
//	////glBindBuffer(GL_SHADER_STORAGE_BUFFER, bvh->bvhCodesSSBO);
//	////	GLuint64 *ptr7 = (GLuint64 *)glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_READ_ONLY);
//	////	for (int i = 0; i < bvh->bufferSizeIn - 1; i++){
//	////
//	////		if (ptr7[i]>=ptr7[i+1]){
//	////			std::cout << ptr7[i] << ", " << ptr7[i + 1] << std::endl;
//	////		}
//	////		std::cout << ptr7[i] << std::endl;
//	////	}
//	////	glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
//	////	std::cout << "" << std::endl; 
//	////	std::cout << "" << std::endl;
//
//
//	////glBindBuffer(GL_SHADER_STORAGE_BUFFER, bvh->BVHAtomics);
//	////glm::uvec2 *ptr = (glm::uvec2 *)glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_READ_ONLY);
//	////for (int i = 0; i < bvh->bufferSizeIn-1; i++){
//	////	glm::uvec2 newSetCount = ptr[i];
//	////	std::cout << newSetCount.x << ", ";
//	////	std::cout << newSetCount.y << ", ";
//	////	std::cout << "" << std::endl;
//	////}
//	////glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
//	////std::cout << "" << std::endl;
//	////std::cout << "" << std::endl;
//
//	////glBindBuffer(GL_SHADER_STORAGE_BUFFER, bvh->bvhTextureCoordsSSBO);
//	////glm::vec4 *ptr8 = (glm::vec4 *)glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_READ_ONLY);
//	////for (int i = 0; i < bvh->bufferSizeIn ; i++){
//	////	glm::vec4 newSetCount = ptr8[i];
//	////	std::cout << newSetCount.x << ", ";
//	////	std::cout << newSetCount.y << ", ";
//	////	std::cout << newSetCount.z << ", ";
//	////	std::cout << newSetCount.w << ", ";
//	////	std::cout << "" << std::endl;
//	////}
//	////std::cout << "Next Set" << std::endl;
//	////glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
//
//
//	//}
//	//traceCounter++;
//	//shaderCompute->stopUsing();
//
//	//shaderComputeColor->use();
//	//
//
//
//	//shaderComputeColor->setUniform(continuousUniform, continuous);
//	//shaderComputeColor->setUniform(samplesUniform3, samples);
//	//shaderComputeColor->setUniform(counterUniform3, traceCounter);
//	//glBindImageTexture(5, display, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA32F);
//	//glDispatchCompute( screenSize.x, screenSize.y, 1 );
//	//glMemoryBarrier( GL_ALL_BARRIER_BITS);
//	//shaderComputeColor->stopUsing();
//
//	//	}
//	//void PTracer::Draw(){
//	//shader->use();
//	//glBindVertexArray(vao);
//	//shader->setUniform(texUniform, 0);
//	//shader->setUniform(cameraUniform, glm::ortho(0.0f, 512.0f, 0.0f, 512.0f, -2.0f, 2.0f));
//	//shader->setUniform(modelUniform, position);
//	//glActiveTexture(GL_TEXTURE0);
//	//glBindTexture(GL_TEXTURE_2D,display);
//	//glDrawElements(GL_TRIANGLES, (6), GL_UNSIGNED_INT, (GLvoid*)0);
//	//glBindVertexArray(0);
//	//shader->stopUsing();
//	}
//void  PTracer::SwitchRender(){
//	if (glfwGetTime()>startSwitch){
//			startSwitch = glfwGetTime() + 1;
//			continuous = !continuous;
//			traceCounter = 0;
//		}
//}
//GLuint PTracer::createTex(){
//	GLuint texName;
//	/*glGenTextures(1, &texName);    
//	glBindTexture(GL_TEXTURE_2D, texName);
//	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
//	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
//	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
//	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
//	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, screenSize.x, screenSize.y, 0, GL_RGBA, GL_FLOAT, NULL);*/
//	return texName;
//}
//
////shading::ShaderSupport*  PTracer::LoadShaders(const char* vertFilename, const char* controlFilename, const char* evaluationFilename, const char* geometryFilename, const char* fragFilename) {
////	std::vector<shading::Shader> shaders;
////	shaders.push_back(shading::Shader::shaderFromFile(vertFilename, GL_VERTEX_SHADER));
////	shaders.push_back(shading::Shader::shaderFromFile(controlFilename, GL_TESS_CONTROL_SHADER));
////	shaders.push_back(shading::Shader::shaderFromFile(evaluationFilename, GL_TESS_EVALUATION_SHADER));
////	shaders.push_back(shading::Shader::shaderFromFile(geometryFilename, GL_GEOMETRY_SHADER));
////	shaders.push_back(shading::Shader::shaderFromFile(fragFilename, GL_FRAGMENT_SHADER));
////	return new shading::ShaderSupport(shaders);
////}
////
////shading::ShaderSupport*  PTracer::LoadShaders(const char* vertFilename, const char* fragFilename) {
////	std::vector<shading::Shader> shaders;
////	shaders.push_back(shading::Shader::shaderFromFile(vertFilename, GL_VERTEX_SHADER));
////	shaders.push_back(shading::Shader::shaderFromFile(fragFilename, GL_FRAGMENT_SHADER));
////	return new shading::ShaderSupport(shaders);
////}
////shading::ShaderSupport*  PTracer::LoadShaders(const char* compFilename) {
////	std::vector<shading::Shader> shaders;
////	shaders.push_back(shading::Shader::shaderFromFile(compFilename, GL_COMPUTE_SHADER));
////	return new shading::ShaderSupport(shaders);
////}
////shading::ShaderSupport*  PTracer::LoadShaders(const char* vertFilename, const char* geometryFilename, const char* fragFilename) {
////	std::vector<shading::Shader> shaders;
////	shaders.push_back(shading::Shader::shaderFromFile(vertFilename, GL_VERTEX_SHADER));
////	shaders.push_back(shading::Shader::shaderFromFile(geometryFilename, GL_GEOMETRY_SHADER));
////	shaders.push_back(shading::Shader::shaderFromFile(fragFilename, GL_FRAGMENT_SHADER));
////	return new shading::ShaderSupport(shaders);
////}