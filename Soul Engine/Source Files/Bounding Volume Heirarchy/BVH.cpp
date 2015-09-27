//#include "BVH.h"
//#include <bitset>
//
//	
//BVH::BVH(){
//	//bvhEqual = LoadShaders("N:\\Documents\\Dropbox\\Phasma\\Phasma-1.0\\Phasma-1.0\\Resource Files\\compute-shader[BVHEqual].txt");
//	//sizeUniform = bvhEqual->uniform("size");
//	//bvhSetup = LoadShaders("N:\\Documents\\Dropbox\\Phasma\\Phasma-1.0\\Phasma-1.0\\Resource Files\\compute-shader[BVHSetup].txt");
//	//sizeUniform5 = bvhSetup->uniform("size");
//
//	//bvhLeafCreation = LoadShaders("N:\\Documents\\Dropbox\\Phasma\\Phasma-1.0\\Phasma-1.0\\Resource Files\\compute-shader[BVHLeafCreation].txt");
//	//objectNodeUniform1 = bvhLeafCreation->uniform("nodeAmount");
//
//
//	//bvhLeafCreationPhysics = LoadShaders("N:\\Documents\\Dropbox\\Phasma\\Phasma-1.0\\Phasma-1.0\\Resource Files\\compute-shader[BVHLeafCreationPhysics].txt");
//	//objectNodeUniform4 = bvhLeafCreationPhysics->uniform("nodeAmount");
//	//timeStep = bvhLeafCreationPhysics->uniform("deltaTime");
//
//
//	//bvhNodeCreation = LoadShaders("N:\\Documents\\Dropbox\\Phasma\\Phasma-1.0\\Phasma-1.0\\Resource Files\\compute-shader[BVHNodeCreation].txt");
//	//objectNodeUniform2 = bvhNodeCreation->uniform("nodeAmount");
//
//	//bvhBoxCreation = LoadShaders("N:\\Documents\\Dropbox\\Phasma\\Phasma-1.0\\Phasma-1.0\\Resource Files\\compute-shader[BVHBoxCreation].txt");
//	//objectNodeUniform3 = bvhBoxCreation->uniform("nodeAmount");
//
//	//shaderGL = LoadShaders("N:\\Documents\\Dropbox\\Phasma\\Phasma-1.0\\Phasma-1.0\\Resource Files\\vertex-shader[bvh].txt", 
//	//	"N:\\Documents\\Dropbox\\Phasma\\Phasma-1.0\\Phasma-1.0\\Resource Files\\fragment-shader[bvh].txt");
//	//cameraUniform = shaderGL->uniform("camera");
//	//lightUniform = shaderGL->uniform("lightNormal");
//
//	//bvhIndexAdd = LoadShaders("N:\\Documents\\Dropbox\\Phasma\\Phasma-1.0\\Phasma-1.0\\Resource Files\\compute-shader[BVHIndices].txt");
//	//sizeUniform4 = bvhIndexAdd->uniform("size");
//	//objectAmountUniform = bvhIndexAdd->uniform("objectAmount");
//
//	//physicsShader = LoadShaders("N:\\Documents\\Dropbox\\Phasma\\Phasma-1.0\\Phasma-1.0\\Resource Files\\compute-shader[BVHPhysics].txt");
//	//sizeUniform6 = physicsShader->uniform("nodeSize");
//
//	//collisionShader = LoadShaders("N:\\Documents\\Dropbox\\Phasma\\Phasma-1.0\\Phasma-1.0\\Resource Files\\compute-shader[CollisionNarrow].txt");
//	//sizeUniform7 = collisionShader->uniform("size");
//	//timeStep2 = collisionShader->uniform("dt");
//	//epsilonS = collisionShader->uniform("EPSILON");
//	////piS = collisionShader->uniform("PI");
//
//	//responseShader = LoadShaders("N:\\Documents\\Dropbox\\Phasma\\Phasma-1.0\\Phasma-1.0\\Resource Files\\compute-shader[PhysicsResponse].txt");
//	//sizeUniform8 = responseShader->uniform("size");
//	//timeStep3 = responseShader->uniform("dt");
//
//
//	bufferSizeVert=0;
//	bufferSizeIn=0;
//	vertexSize = 0;
//	indexSize = 0;
//
//	/*vertexAttrib = shaderGL->attrib("vert_VS_in");
//	texCoordAttrib = shaderGL->attrib("texCoord_VS_in");
//	normalAttrib = shaderGL->attrib("normal_VS_in");
//
//	glGenBuffers(1, &bvhVerticesSSBO);
//	glGenBuffers(1, &bvhIndicesSSBO);
//	glGenBuffers(1, &bvhNormalsSSBO);
//	glGenBuffers(1, &bvhTextureCoordsSSBO);
//	glGenBuffers(1, &vertices);
//	glGenBuffers(1, &indices);
//	glGenBuffers(1, &texUBO);
//	glGenBuffers(1, &bvhCodesSSBO);
//	glGenBuffers(1, &BVHStructure);
//	glGenBuffers(1, &BVHAtomics);
//	glGenBuffers(1, &materialsSSBO);
//	glGenBuffers(1, &physicsSSBO);
//	glGenBuffers(1, &collisionsSSBO);
//	glGenBuffers(1, &collisionAtomic);
//	glGenBuffers(1, &collisionTestSSBO);
//	glGenBuffers(1, &responseAtomic);
//	glGenBuffers(1, &responsesSSBO);
//	glGenBuffers(1, &responsesExtraSSBO);
//	glGenBuffers(1, &physicsNewSSBO);
//	glGenVertexArrays(1, &VAO);*/
//
//}
//void BVH::Add(Object* ob){
//	storage.push_back(ob);
//}
//void BVH::ExtractObjects(){
//	
//
//	//bvhVertices.resize(storage.size());
//	//SizeVertices.resize(storage.size());
//	//bvhIndices.resize(storage.size());
//	//SizeIndices.resize(storage.size());
//	//bvhTextureCoords.resize(storage.size());
//	//bvhNormals.resize(storage.size());
//
//	//int i = 0;
//	//for (std::list<Object*>::iterator itr = storage.begin(); itr != storage.end(); itr ++,i++ ){
//	//	if ((*itr)->loaded){
//	//
//	//		(*itr)->GetVertices(bvhVertices[i], SizeVertices[i]);
//
//	//		(*itr)->GetIndices(bvhIndices[i], SizeIndices[i]);
//
//	//		GLuint dummy;
//	//		(*itr)->GetTextureCoords(bvhTextureCoords[i], dummy);
//
//	//		(*itr)->GetNormals(bvhNormals[i], dummy);
//
//	//		(*itr)->GetMaterials(materials);
//	//	}
//	//}
//
//	//texHandles.resize(materials.size());
//	//bvhMaterials.resize(textureSize);
//
//	//GLuint indexPos = 0;
//	//i = 0;
//	//for (std::list<Material*>::iterator itr = materials.begin(); itr != materials.end(); itr++,i++){
//	//	//check_gl_error();
//	//	texHandles[i] = ((*itr)->GetHandle());
//	//	check_gl_error();                 //handles not working properly
//	//	bvhMaterials[i].texturePos = i;
//
//	//	bvhMaterials[i].startIndex = indexPos;
//
//	//	indexPos += SizeVertices[i];
//
//	//	bvhMaterials[i].endIndexExclusive = indexPos;
//	//}
//	//
//	//for (int i = 0; i<textureSize; i++){
//	//	if (i < texHandles.size()){
//	//		textures[i] = texHandles[i];
//	//	}
//	//	else{
//	//		textures[i] = 0;
//
//	//		bvhMaterials[i].startIndex = 0;
//	//		bvhMaterials[i].texturePos = i;
//	//		bvhMaterials[i].endIndexExclusive = indexPos;
//	//	}
//	//}
//}
//void BVH::UpdateObjects(double dt){
//	for (std::list<Object*>::iterator itr = storage.begin(); itr != storage.end(); itr++){
//		if ((*itr)->loaded){
//			(*itr)->Update((float)dt);
//		}
//		if (!(*itr)->loaded){
//			(*itr)->Load();
//			(*itr)->Update((float)dt);
//			//threads->enqueue(std::bind(&((*itr)->Load), (itr)));
//		}
//	}
//}
//
//void BVH::Physics(double dt){
//
//		//std::vector<Properties> phyPositions;
//
//		//for (std::list<Object*>::iterator itr = storage.begin(); itr != storage.end(); itr++){
//		//	if ((*itr)->loaded){
//		//		glm::vec3 tempP;
//		//		glm::vec3 tempV;
//		//		bool tempStat;
//		//		(*itr)->GetPhysics(tempP, tempV, tempStat);
//
//		//		if (!tempStat){
//		//			tempV.y += (GRAVITY*METER*float(dt));
//		//		}
//
//		//		float tempStatF;
//		//		if (tempStat){
//		//			tempStatF = 1.0f;
//		//		}
//		//		else{
//		//			tempStatF = 0.0f;
//		//		}
//
//		//		Properties temp = { glm::vec4(tempP.x, tempP.y, tempP.z, tempStatF), glm::vec4(tempV.x, tempV.y, tempV.z, 1.0f), glm::uvec4(1000000000, 1000000000, 1000000000, 1000000000) };
//		//		phyPositions.push_back(temp);
//		//	}
//		//}
//
//		//glBindBuffer(GL_SHADER_STORAGE_BUFFER, physicsSSBO);
//		//glBufferData(GL_SHADER_STORAGE_BUFFER,
//		//	phyPositions.size()*sizeof(Properties),
//		//	phyPositions.data(), GL_STATIC_DRAW);
//		//glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
//
//		//glBindBuffer(GL_SHADER_STORAGE_BUFFER, physicsNewSSBO);
//		//glBufferData(GL_SHADER_STORAGE_BUFFER,
//		//	phyPositions.size()*sizeof(Properties),
//		//	phyPositions.data(), GL_STATIC_DRAW);
//		//glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
//
//		//glBindBuffer(GL_SHADER_STORAGE_BUFFER, bvhCodesSSBO);
//		//glBufferData(GL_SHADER_STORAGE_BUFFER,
//		//	bufferSizeIn*sizeof(GLuint64),
//		//	NULL, GL_STATIC_DRAW);
//		//glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
//
//		//glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, bvhVerticesSSBO);
//		//glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, bvhIndicesSSBO);
//		//glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, bvhCodesSSBO);
//
//
//		//GLuint dispatchSize = (bufferSizeIn / 64) + ((bufferSizeIn % 64 == 0) ? 0 : 1);
//
//		//bvhSetup->use();
//		//bvhSetup->setUniform(sizeUniform5, bufferSizeIn);
//		//glDispatchCompute(dispatchSize, 1, 1);
//		//glMemoryBarrier(GL_ALL_BARRIER_BITS);
//
//		//bvhSetup->stopUsing();
//
//		//double currentTime = glfwGetTime();
//
//		//sorterRadix->Sort(bvhCodesSSBO, bvhIndicesSSBO, bufferSizeIn, 64);
//
//		//currentTime = glfwGetTime() - currentTime;
//
//		//std::cout <<"Sorting time (ms): "<< (float)currentTime*1000 << std::endl;
//
//		//CreateHeirarchy(true,dt);
//
//
//		//GLuint zero = 0;
//
//		//glBindBuffer(GL_SHADER_STORAGE_BUFFER, collisionAtomic);
//		//glBufferData(GL_SHADER_STORAGE_BUFFER,
//		//	sizeof(GLuint),
//		//	&zero, GL_STATIC_DRAW);
//		//glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
//
//
//		//glBindBuffer(GL_SHADER_STORAGE_BUFFER, collisionsSSBO);
//		//glBufferData(GL_SHADER_STORAGE_BUFFER,
//		//	((bufferSizeIn*bufferSizeIn)*0.025 * 15)*sizeof(glm::uvec4),
//		//	NULL, GL_STATIC_DRAW);
//		//glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
//
//		//glBindBuffer(GL_SHADER_STORAGE_BUFFER, collisionTestSSBO);
//		//glBufferData(GL_SHADER_STORAGE_BUFFER,
//		//	((bufferSizeIn*bufferSizeIn)*0.025 * 15)*sizeof(glm::uvec4),
//		//	NULL, GL_STATIC_DRAW);
//		//glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
//
//		//glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, collisionAtomic);
//		//glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, collisionsSSBO);
//		//glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, bvhIndicesSSBO);
//		//glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, BVHStructure);
//		//glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, BVHAtomics);
//		//glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, collisionTestSSBO);
//
//		//dispatchSize = (bufferSizeIn / 32) + ((bufferSizeIn % 32 == 0) ? 0 : 1);
//		//
//		//physicsShader->use();
//		//physicsShader->setUniform(sizeUniform6, bufferSizeIn);
//		//glDispatchCompute(dispatchSize, 1, 1);
//		//glMemoryBarrier(GL_ALL_BARRIER_BITS);
//		//physicsShader->stopUsing();
//
//		//currentTime = glfwGetTime();
//
//		//physicsShader->use();
//		//physicsShader->setUniform(sizeUniform6, bufferSizeIn);
//		//glDispatchCompute(dispatchSize, 1, 1);
//		//glMemoryBarrier(GL_ALL_BARRIER_BITS);
//		//physicsShader->stopUsing();
//
//		//currentTime = glfwGetTime() - currentTime;
//
//		//std::cout << "     BVH Collision(ms): " << currentTime * 1000 << std::endl;
//
//		//glBindBuffer(GL_SHADER_STORAGE_BUFFER, collisionAtomic);
//		//GLuint *ptr = (GLuint *)glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_READ_ONLY);
//		//GLuint  size = ptr[0];
//		//glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
//	
//
//		//glBindBuffer(GL_SHADER_STORAGE_BUFFER, responseAtomic);
//		//glBufferData(GL_SHADER_STORAGE_BUFFER,
//		//	sizeof(GLuint),
//		//	&zero, GL_STATIC_DRAW);
//		//glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
//
//		//glBindBuffer(GL_SHADER_STORAGE_BUFFER, responsesSSBO);
//		//glBufferData(GL_SHADER_STORAGE_BUFFER,
//		//	((bufferSizeIn*bufferSizeIn)*0.025)*sizeof(glm::vec4),
//		//	NULL, GL_STATIC_DRAW);
//		//glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
//
//		//glBindBuffer(GL_SHADER_STORAGE_BUFFER, responsesExtraSSBO);
//		//glBufferData(GL_SHADER_STORAGE_BUFFER,
//		//	((bufferSizeIn*bufferSizeIn)*0.025)*sizeof(glm::uvec2),
//		//	NULL, GL_STATIC_DRAW);
//		//glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
//
//		//dispatchSize = (size / 64) + ((size % 64 == 0) ? 0 : 1);
//
//		//glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, collisionsSSBO);
//		//glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, bvhIndicesSSBO);
//		//glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, vertices);
//		//glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, physicsSSBO);
//		//glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, bvhVerticesSSBO);
//		//glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, collisionTestSSBO);
//		//glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 7, responseAtomic);
//		//glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 8, responsesSSBO);
//		//glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 9, responsesExtraSSBO);
//
//
//
//
//		//collisionShader->use();
//		//collisionShader->setUniform(sizeUniform7, size);
//		//collisionShader->setUniform(timeStep2, float(dt));
//		//collisionShader->setUniform(epsilonS, EPSILON);
//		////collisionShader->setUniform(piS, PI);
//		//glDispatchCompute(dispatchSize, 1, 1);
//		//glMemoryBarrier(GL_ALL_BARRIER_BITS);
//		//collisionShader->stopUsing();
//
//		//currentTime = glfwGetTime();
//
//		//collisionShader->use();
//		//collisionShader->setUniform(sizeUniform7, size);
//		//collisionShader->setUniform(timeStep2, float(dt));
//		//collisionShader->setUniform(epsilonS, EPSILON);
//		////collisionShader->setUniform(piS, PI);
//		//glDispatchCompute(dispatchSize, 1, 1);
//		//glMemoryBarrier(GL_ALL_BARRIER_BITS);
//		//collisionShader->stopUsing();
//
//		//currentTime = glfwGetTime() - currentTime;
//
//		//std::cout << "     Narrow Collision(ms): " << currentTime * 1000 << std::endl;
//
//
//
//		//glBindBuffer(GL_SHADER_STORAGE_BUFFER, responseAtomic);
//		//GLuint *ptr1 = (GLuint *)glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_READ_ONLY);
//		//GLuint  resSize = ptr1[0];
//		//glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
//
//
//
//
//		//glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, responsesSSBO);
//		//glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, collisionsSSBO);
//		//glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, bvhIndicesSSBO);
//		//glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, physicsSSBO);
//		//glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, responsesExtraSSBO);
//		//glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, bvhVerticesSSBO);
//		//glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 7, physicsNewSSBO);
//
//		//dispatchSize = (resSize / 64) + ((resSize % 64 == 0) ? 0 : 1);
//
//
//		//responseShader->use();
//		//responseShader->setUniform(sizeUniform8, resSize);
//		//responseShader->setUniform(timeStep3, float(dt));
//		//glDispatchCompute(dispatchSize, 1, 1);
//		//glMemoryBarrier(GL_ALL_BARRIER_BITS);
//		//responseShader->stopUsing();
//
//		//currentTime = glfwGetTime();
//
//		//responseShader->use();
//		//responseShader->setUniform(sizeUniform8, resSize);
//		//responseShader->setUniform(timeStep3, float(dt));
//		//glDispatchCompute(dispatchSize, 1, 1);
//		//glMemoryBarrier(GL_ALL_BARRIER_BITS);
//		//responseShader->stopUsing();
//
//		//currentTime = glfwGetTime() - currentTime;
//
//		//std::cout << "     Response(ms): " << currentTime * 1000 << std::endl;
//
//		//glBindBuffer(GL_SHADER_STORAGE_BUFFER, physicsNewSSBO);
//		//glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, phyPositions.size()*sizeof(Properties), phyPositions.data());
//		//glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
//
//		//int i = 0;
//		//for (std::list<Object*>::iterator itr = storage.begin(); itr != storage.end(); itr++,i++){
//		//	if ((*itr)->loaded){
//		//		glm::vec3 tempP = glm::vec3(phyPositions[i].position.x, phyPositions[i].position.y, phyPositions[i].position.z);
//		//		glm::vec3 tempV = glm::vec3(phyPositions[i].velocity.x, phyPositions[i].velocity.y, phyPositions[i].velocity.z);
//		//		glm::uvec4 tempData = phyPositions[i].data;
//
//		//		float tempStatF = phyPositions[i].position.w;
//		//		bool tempStat;
//		//		if (tempStatF==1.0f){
//		//			tempStat = true;
//		//		}
//		//		else{
//		//			tempStat = false;
//		//		}
//		//		if (!tempStat){
//		//			tempP += tempV*glm::vec3(float(dt))* glm::vec3(tempData.x / 1000000000.0f);
//		//		}
//		//		(*itr)->SetPhysics(tempP, tempV, tempStat);
//		//	}
//		//}
//
//		//
//
//
//}
//void BVH::SetupObjects(){
//	
////	ExtractObjects();
////	
////
////	bufferSizeVert = 0;
////	bufferSizeIn = 0;
////	for (GLuint i = 0; i < bvhVertices.size(); i++){
////		bufferSizeVert += SizeVertices[i];
////	}
////	for (GLuint i = 0; i < bvhIndices.size(); i++){
////		bufferSizeIn += SizeIndices[i];
////	}
////	
////
////	glBindBuffer(GL_SHADER_STORAGE_BUFFER, materialsSSBO);
////	glBufferData(GL_SHADER_STORAGE_BUFFER,
////		textureSize*sizeof(bvhMat),
////		bvhMaterials.data(), GL_STATIC_DRAW);
////	glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
////
////	glBindBuffer(GL_SHADER_STORAGE_BUFFER, bvhVerticesSSBO);
////	glBufferData(GL_SHADER_STORAGE_BUFFER,
////		bufferSizeVert*sizeof(glm::vec4),
////		NULL, GL_STATIC_DRAW);
////	glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
////	
////
////	glBindBuffer(GL_SHADER_STORAGE_BUFFER, bvhIndicesSSBO);
////	glBufferData(GL_SHADER_STORAGE_BUFFER,
////		bufferSizeIn*sizeof(glm::uvec3),
////		NULL, GL_STATIC_DRAW);
////	glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
////
////	
////	glBindBuffer(GL_SHADER_STORAGE_BUFFER, bvhNormalsSSBO);
////	glBufferData(GL_SHADER_STORAGE_BUFFER,
////		bufferSizeVert*sizeof(glm::vec4),
////		NULL, GL_STATIC_DRAW);
////	glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
////
////	
////	glBindBuffer(GL_SHADER_STORAGE_BUFFER, bvhTextureCoordsSSBO);
////	glBufferData(GL_SHADER_STORAGE_BUFFER,
////		bufferSizeVert*sizeof(glm::vec4),
////		NULL, GL_STATIC_DRAW);
////	glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
////
////	
////	glBindBuffer(GL_SHADER_STORAGE_BUFFER, vertices);
////	glBufferData(GL_SHADER_STORAGE_BUFFER,
////		SizeVertices.size()*sizeof(GLuint),
////		SizeVertices.data(), GL_STATIC_DRAW);
////	glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
////	
////
////	glBindBuffer(GL_SHADER_STORAGE_BUFFER, indices);
////	glBufferData(GL_SHADER_STORAGE_BUFFER,
////		SizeIndices.size()*sizeof(GLuint),
////		SizeIndices.data(), GL_STATIC_DRAW);
////	glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
////
////	
////	glBindBuffer(GL_UNIFORM_BUFFER, texUBO);
////	glBufferData(GL_UNIFORM_BUFFER, sizeof(textures), &textures[0], GL_STATIC_DRAW);
////	glBindBuffer(GL_UNIFORM_BUFFER, 0);
////
////	GLuint amount = 0;
////	for (GLuint i = 0; i <storage.size(); i++){
////
////		glBindBuffer(GL_COPY_READ_BUFFER, bvhVertices[i]);
////		glBindBuffer(GL_COPY_WRITE_BUFFER, bvhVerticesSSBO);
////		glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, 0, amount, SizeVertices[i] * sizeof(glm::vec4));
////		glBindBuffer(GL_COPY_WRITE_BUFFER, 0);
////		glBindBuffer(GL_COPY_READ_BUFFER, 0);
////		//check_gl_error();
////
////		glBindBuffer(GL_COPY_READ_BUFFER, bvhTextureCoords[i]);
////		glBindBuffer(GL_COPY_WRITE_BUFFER, bvhTextureCoordsSSBO);
////		glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, 0, amount, SizeVertices[i] * sizeof(glm::vec4));
////		glBindBuffer(GL_COPY_WRITE_BUFFER, 0);
////		glBindBuffer(GL_COPY_READ_BUFFER, 0);
////
////		//check_gl_error();
////		glBindBuffer(GL_COPY_READ_BUFFER, bvhNormals[i]);
////		glBindBuffer(GL_COPY_WRITE_BUFFER, bvhNormalsSSBO);
////		glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, 0, amount, SizeVertices[i] * sizeof(glm::vec4));
////		glBindBuffer(GL_COPY_WRITE_BUFFER, 0);
////		glBindBuffer(GL_COPY_READ_BUFFER, 0);
////		//check_gl_error();
////		amount += SizeVertices.at(i)*sizeof(glm::vec4);
////	}
////	amount = 0;
////	
////	for (std::vector<int>::size_type i = 0; i != bvhIndices.size(); i++){
////
////		glBindBuffer(GL_COPY_READ_BUFFER, bvhIndices[i]);
////		glBindBuffer(GL_COPY_WRITE_BUFFER, bvhIndicesSSBO);
////		glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, 0, amount, SizeIndices[i] * sizeof(glm::uvec3));
////		glBindBuffer(GL_COPY_WRITE_BUFFER, 0);
////		glBindBuffer(GL_COPY_READ_BUFFER, 0);
////
////		amount += SizeIndices.at(i)*sizeof(glm::uvec3);
////	}
////	
////
////	//glBindBuffer(GL_SHADER_STORAGE_BUFFER, bvhVerticesSSBO);
////	//	glm::vec4 *ptr2 = (glm::vec4 *)glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_READ_ONLY);
////	//	for (int i = 0; i < bufferSizeVert;i++){
////	//		std::cout << ptr2[i].x << ", " << ptr2[i].y << ", " << ptr2[i].z << std::endl;
////	//	}
////	//	glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
////
////	indexSize = (GLuint)SizeIndices.size();
////	vertexSize = (GLuint)SizeVertices.size();
////	SizeVertices.clear();
////	SizeIndices.clear();
////	bvhVertices.clear();
////	bvhNormals.clear();
////	materials.clear();
////	bvhMaterials.clear();
////	bvhIndices.clear();
////	bvhTextureCoords.clear();
////	texHandles.clear();
////	
////
////
////
////	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, bvhIndicesSSBO);
////	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, vertices);
////	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, indices);
////
////	GLuint dispatchSize = (bufferSizeIn / 64) + ((bufferSizeIn % 64 == 0) ? 0 : 1);
////
////	bvhIndexAdd->use();
////
////	bvhIndexAdd->setUniform(objectAmountUniform, GLuint(indexSize) - 1);
////	bvhIndexAdd->setUniform(sizeUniform4, bufferSizeIn);
////	glDispatchCompute(dispatchSize, 1, 1);
////	glMemoryBarrier(GL_ALL_BARRIER_BITS);
////
////	bvhIndexAdd->stopUsing();
////}
////
////
////void BVH::CreateHeirarchy(bool runPhys,double dt){
////	
////	nodeAmount = bufferSizeIn;
////
////	GLuint dispatchSize = (nodeAmount / 64) + ((nodeAmount % 64 == 0) ? 0 : 1);
////	
////	glBindBuffer(GL_SHADER_STORAGE_BUFFER, BVHStructure);
////	glBufferData(GL_SHADER_STORAGE_BUFFER,
////		((nodeAmount - 1) * 2 * sizeof(glm::vec4)) + (nodeAmount* 3 * sizeof(glm::vec4)),
////		NULL, GL_STATIC_DRAW);
////	glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
////	
////	
////	glBindBuffer(GL_SHADER_STORAGE_BUFFER, BVHAtomics);
////	glBufferData(GL_SHADER_STORAGE_BUFFER,
////		(nodeAmount - 1) * sizeof(glm::uvec2),
////		NULL, GL_STATIC_DRAW);
////	glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
////
////	
////	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 10, bvhVerticesSSBO);
////	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 11, bvhIndicesSSBO);
////	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 12, vertices);
////	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 13, indices);
////	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 14, bvhCodesSSBO);
////	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 15, BVHStructure);
////	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 16, BVHAtomics);
////
////	
////	
////	
////
////	if (runPhys){
////		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 17, physicsSSBO);
////		bvhLeafCreationPhysics->use();
////		bvhLeafCreationPhysics->setUniform(timeStep, float(dt));
////		bvhLeafCreationPhysics->setUniform(objectNodeUniform4, nodeAmount);
////		glDispatchCompute(dispatchSize, 1, 1);
////		glMemoryBarrier(GL_ALL_BARRIER_BITS);
////
////		bvhLeafCreationPhysics->stopUsing();
////	}
////	else{
////		bvhLeafCreation->use();
////		bvhLeafCreation->setUniform(objectNodeUniform1, nodeAmount);
////		glDispatchCompute(dispatchSize, 1, 1);
////		glMemoryBarrier(GL_ALL_BARRIER_BITS);
////
////		bvhLeafCreation->stopUsing();
////	}
////
////	nodeAmount = nodeAmount - 1;
////
////	dispatchSize = (nodeAmount / 64) + ((nodeAmount % 64 == 0) ? 0 : 1);
////
////	
////	bvhNodeCreation->use();
////	bvhNodeCreation->setUniform(objectNodeUniform2, nodeAmount);
////	glDispatchCompute(dispatchSize, 1, 1);
////	glMemoryBarrier(GL_ALL_BARRIER_BITS);
////
////	bvhNodeCreation->stopUsing();
////
////
////	nodeAmount = nodeAmount +1;
////
////
////
////
////
////	dispatchSize = (nodeAmount / 64) + ((nodeAmount % 64 == 0) ? 0 : 1);
////
////	bvhBoxCreation->use();
////	bvhBoxCreation->setUniform(objectNodeUniform3, nodeAmount);
////	glDispatchCompute(dispatchSize, 1, 1);
////	glMemoryBarrier(GL_ALL_BARRIER_BITS);
////
////	bvhBoxCreation->stopUsing();
////
//}
//
//void BVH::Draw(Camera& camera){
//	////check_gl_error();
//	//	glBindVertexArray(VAO);
//
//	//	glBindBuffer(GL_ARRAY_BUFFER, bvhVerticesSSBO);
//	//	glEnableVertexAttribArray(vertexAttrib);
//	//	glVertexAttribPointer(vertexAttrib, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(GLfloat), NULL);
//
//	//	glBindBuffer(GL_ARRAY_BUFFER, bvhTextureCoordsSSBO);
//	//	glEnableVertexAttribArray(texCoordAttrib);
//	//	glVertexAttribPointer(texCoordAttrib, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(GLfloat), NULL);
//
//	//	glBindBuffer(GL_ARRAY_BUFFER, bvhNormalsSSBO);
//	//	glEnableVertexAttribArray(normalAttrib);
//	//	glVertexAttribPointer(normalAttrib, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(GLfloat), NULL);
//
//
//	//	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, bvhIndicesSSBO);
//
//	//	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, materialsSSBO);
//	//	glBindBufferBase(GL_UNIFORM_BUFFER, 2, texUBO);
//
//	//	glBindVertexArray(0);
//
//	//	shaderGL->use();
//	//	shaderGL->setUniform(lightUniform, 0.0f, 1.0f, 0.0f);
//	//	shaderGL->setUniform(cameraUniform, camera.matrix());
//	//
//
//	//	glBindVertexArray(VAO);
//	//	glDrawElements(GL_TRIANGLES, 3*bufferSizeIn, GL_UNSIGNED_INT, (GLvoid*)0);
//	//	//check_gl_error();
//
//	//	glBindVertexArray(0);
//
//	//	shaderGL->stopUsing();
//
//}
//void BVH::Clean(){
//	//glDeleteBuffers(1, &vertices);
//	//glDeleteBuffers(1, &indices);
//	//glDeleteBuffers(1, &texUBO);
//	//glDeleteBuffers(1, &bvhVerticesSSBO);
//	//glDeleteBuffers(1, &bvhIndicesSSBO);
//	//glDeleteBuffers(1, &bvhCodesSSBO);
//	//glDeleteBuffers(1, &bvhNormalsSSBO);
//	//glDeleteBuffers(1, &BVHStructure);
//	//glDeleteBuffers(1, &BVHAtomics);
//	//glDeleteBuffers(1, &physicsSSBO);
//	//glDeleteBuffers(1, &bvhTextureCoordsSSBO);
//	//glDeleteBuffers(1, &materialsSSBO);
//	//glDeleteBuffers(1, &collisionsSSBO);
//	//glDeleteBuffers(1, &collisionAtomic);
//	//glDeleteBuffers(1, &collisionTestSSBO);
//	//glDeleteBuffers(1, &responseAtomic);
//	//glDeleteBuffers(1, &responsesSSBO);
//	//glDeleteBuffers(1, &responsesExtraSSBO);
//	//glDeleteBuffers(1, &physicsNewSSBO);
//	//glDeleteBuffers(1, &VAO);
//}