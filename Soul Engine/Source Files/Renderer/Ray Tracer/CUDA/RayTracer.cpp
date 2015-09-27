//#include "RayTracer.h"
//
//RayTracer::RayTracer(glm::uvec2 resolution){
//	/*focalDistance = 100;
//	apertureRadius = 4 * MILLIMETER;
//	iterationCounter = 0;
//	GPULocalInvocations = 32;
//	continuous = false;
//	storedScreen = resolution;
//	computeRays = LoadShaders("N:\\Documents\\Dropbox\\Phasma\\Phasma-1.0\\Phasma-1.0\\Resource Files\\compute-shader[RayTracerCompute].txt");
//
//	counterUniform = computeRays->uniform("counter");
//	cameraApertureRadiusUniform = computeRays->uniform("cameraApertureRadius");
//	cameraFocalDistanceUniform = computeRays->uniform("cameraFocalDistance");
//	cameraPositionUniform = computeRays->uniform("cameraPosition");
//	screenUniform = computeRays->uniform("screen");
//	nodeSizeUniform = computeRays->uniform("nodeSize");
//	samplesUniform = computeRays->uniform("samples");
//	horizontalAxisUniform = computeRays->uniform("horizontalAxis");
//	verticalAxisUniform = computeRays->uniform("verticalAxis");
//	middleUniform = computeRays->uniform("middle");
//	horizontalUniform = computeRays->uniform("horizontal");
//	verticalUniform = computeRays->uniform("vertical");
//	continuousUniform = computeRays->uniform("continuous");
//	sizeUniform = computeRays->uniform("size");
//	textureUniforms = glGetUniformBlockIndex(computeRays->object(), "Samplers");
//
//	glGenBuffers(1, &renderBuffer);
//	glBindBuffer(GL_SHADER_STORAGE_BUFFER, renderBuffer);
//	glBufferData(GL_SHADER_STORAGE_BUFFER,
//		storedScreen.x*storedScreen.y*sizeof(glm::vec4),
//		NULL, GL_STATIC_DRAW);
//
//	glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);*/
//}
//
//GLuint RayTracer::Draw(glm::uvec2 screenSize, BVH* bvh, Camera& camera,GLuint samples){
//
//	/*GLuint dispatchThreadsSingle = screenSize.x*screenSize.y;
//	GLuint dispatchThreads = dispatchThreadsSingle*samples;
//	GLuint dispatchCall = (dispatchThreads / GPULocalInvocations) + ((dispatchThreads % GPULocalInvocations == 0) ? 0 : 1);
//
//	if (screenSize != storedScreen){
//		iterationCounter = 0;
//		storedScreen = screenSize;	
//	}
//
//	if (!continuous){
//		glm::vec4 clearColor = glm::vec4(0.0f, 0.0f, 0.0f,1.0f);
//		glClearNamedBufferData(renderBuffer,
//			GL_RGBA32F,
//			GL_RGBA,
//			GL_FLOAT,
//			&clearColor);
//	}
//	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, bvh->bvhVerticesSSBO);
//	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, bvh->bvhIndicesSSBO);
//	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, bvh->bvhNormalsSSBO);
//	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, bvh->bvhTextureCoordsSSBO);
//	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, bvh->BVHStructure);
//	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, bvh->BVHAtomics);
//	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 7, bvh->materialsSSBO);
//	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 8, renderBuffer);
//	glBindBufferBase(GL_UNIFORM_BUFFER, 9, bvh->texUBO);
//
//
//	glm::vec3 horizontalAxis = glm::cross(glm::normalize(camera.forward()), camera.up());
//	horizontalAxis = glm::normalize(horizontalAxis);
//	glm::vec3  verticalAxis = glm::cross(horizontalAxis, glm::normalize(camera.forward()));
//	verticalAxis = glm::normalize(verticalAxis);
//
//	glm::vec3  middle = camera.position() + glm::normalize(camera.forward());
//	glm::vec3  horizontal = horizontalAxis * glm::vec3(glm::tan(glm::radians(camera.fieldOfView().x * 0.5f )));
//	glm::vec3  vertical = verticalAxis * glm::vec3(glm::tan(glm::radians(-(camera.fieldOfView().x / camera.viewportAspectRatio()) * 0.5f )));
//
//
//	computeRays->use();
//	computeRays->setUniform(counterUniform, iterationCounter);
//	computeRays->setUniform(cameraApertureRadiusUniform, apertureRadius);
//	computeRays->setUniform(cameraFocalDistanceUniform, focalDistance);
//	computeRays->setUniform(cameraPositionUniform, camera.position().x, camera.position().y, camera.position().z);
//	computeRays->setUniform(screenUniform, screenSize.x, screenSize.y);
//	computeRays->setUniform(samplesUniform, samples);
//	computeRays->setUniform(horizontalAxisUniform, horizontalAxis.x, horizontalAxis.y, horizontalAxis.z);
//	computeRays->setUniform(verticalAxisUniform, verticalAxis.x, verticalAxis.y, verticalAxis.z);
//	computeRays->setUniform(middleUniform, middle.x, middle.y, middle.z);
//	computeRays->setUniform(horizontalUniform, horizontal.x, horizontal.y, horizontal.z);
//	computeRays->setUniform(verticalUniform, vertical.x, vertical.y, vertical.z);
//	computeRays->setUniform(continuousUniform, continuous);
//	computeRays->setUniform(nodeSizeUniform, bvh->bufferSizeIn);
//	computeRays->setUniform(sizeUniform, dispatchThreads);
//
//	glUniformBlockBinding(computeRays->object(), textureUniforms, 9);
//
//	glDispatchCompute(dispatchCall, 1, 1);
//	glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
//	computeRays->stopUsing();
//
//	iterationCounter++;
//
//	return renderBuffer;*/
//	return 0;
//}