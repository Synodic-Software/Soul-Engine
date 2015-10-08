//#pragma once
//
//#include "Engine Core/BasicDependencies.h"
//#include "Utility/OpenGL/ShaderSupport.h"
//#include "Utility/OpenGL/Shader.h"
//#include "Engine Core/Material/Texture/Texture.h"
//#include "Engine Core/Camera/Camera.cuh"
//#include "Engine Core/Material/Texture/Bitmap.h"
////#include "Algorithms/Data Algorithms/GPU Quicksort/CUDA/QuickSort.h"
//#include "Engine Core/Material/Material.h"
//#include "Engine Core/Object/Object.h"
//
//#define textureSize 10
//
//
//class BVH{
//private: 
//
//shading::ShaderSupport* bvhEqual;
//
//public:
//	std::list<Object*> storage;
//
//	BVH();
//	void SetupObjects(void);	
//	void CreateHeirarchy(bool, double);
//	void Clean(void);
//	void Draw(Camera&); 
//	void ExtractObjects();
//	void LoadObjects();
//	void Physics(double);
//	void UpdateObjects(double);
//	void InitializeObjects();
//
//	void Add(Object*);
//
//	std::vector<GLuint> SizeVertices;
//	std::vector<GLuint> SizeIndices;	
//	std::vector<GLuint> bvhVertices;
//	std::vector<GLuint> bvhNormals;
//	std::vector<GLuint> bvhIndices;
//	std::vector<GLuint> bvhTextureCoords;
//
//	GLuint materialsSSBO;
//	/*GLuint bvhVerticesSSBO;
//	GLuint bvhNormalsSSBO;
//	GLuint bvhIndicesSSBO;
//	GLuint bvhTextureCoordsSSBO;*/
//	GLuint collisionsSSBO;
//	
//	GLuint BVHStructure;
//	GLuint BVHAtomics;
//	GLuint texUBO;
//	GLuint bufferSizeIn;
//	GLuint bvhCodesSSBO;
//	
//	GLuint bufferSizeVert;
//	GLuint indexSize;
//	GLuint vertexSize;
//	GLuint propPointerSSBO;
//
//	GLuint VAO;
//	GLuint64 textures[textureSize];
//
//private:
//	GLuint physicsSSBO;
//	GLuint collisionAtomic;
//
//	shading::ShaderSupport* LoadShaders(const char* compFilename) {
//		std::vector<shading::Shader> shaders;
//		shaders.push_back(shading::Shader::shaderFromFile(compFilename, GL_COMPUTE_SHADER));
//		return new shading::ShaderSupport(shaders);
//	}
//	shading::ShaderSupport* LoadShaders(const char* vertFilename, const char* fragFilename) {
//		std::vector<shading::Shader> shaders;
//		shaders.push_back(shading::Shader::shaderFromFile(vertFilename, GL_VERTEX_SHADER));
//		shaders.push_back(shading::Shader::shaderFromFile(fragFilename, GL_FRAGMENT_SHADER));
//		return new shading::ShaderSupport(shaders);
//	}
//	//struct set{
//	//	GLuint start;
//	//	GLuint end;
//	//	GLuint countLess;
//	//	GLuint countEqual;
//	//	GLuint countGreater;
//	//	GLuint predicate;
//	//};
//
//	typedef struct bvhMat{
//		GLuint texturePos;
//		GLuint startIndex;
//		GLuint endIndexExclusive;
//	}bvhMat;
//
//	typedef struct
//	{
//		glm::vec4 position;
//		glm::vec4 velocity;
//		glm::uvec4 data;
//	} Properties;
//
//
//	std::vector<bvhMat> bvhMaterials;
//	std::list<Material*> materials;
//
//	GLuint vertices;
//	GLuint indices;
//	GLuint nodeAmount;
//
//	GLuint vertexAttrib;
//	GLuint normalAttrib;
//	GLuint texCoordAttrib;
//
//	GLuint piS;
//	GLuint epsilonS;
//	GLuint timeStep2;
//	GLuint timeStep3;
//	GLuint sizeUniform4;
//	GLuint sizeUniform5;
//	GLuint sizeUniform6;
//	GLuint sizeUniform7;
//	GLuint objectAmountUniform;
//	GLuint objectNodeUniform1;
//	GLuint objectNodeUniform2;
//	GLuint objectNodeUniform3;
//	GLuint objectNodeUniform4;
//	GLuint timeStep;
//	GLuint sizeUniform;
//	GLuint lightUniform;
//	GLuint cameraUniform;
//	GLuint collisionTestSSBO;
//	GLuint sizeUniform8;
//	GLuint responseAtomic;
//	GLuint responsesSSBO;
//	GLuint responsesExtraSSBO;
//	GLuint physicsNewSSBO;
//
//	shading::ShaderSupport* bvhIndexAdd;
//	shading::ShaderSupport* bvhSetup;
//	
//	shading::ShaderSupport* bvhLeafCreation;
//	shading::ShaderSupport* bvhLeafCreationPhysics;
//	shading::ShaderSupport* bvhNodeCreation;
//	shading::ShaderSupport* bvhBoxCreation;
//	shading::ShaderSupport* physicsShader;
//	shading::ShaderSupport* collisionShader;
//	shading::ShaderSupport* responseShader;
//	
//    std::vector<GLuint64> texHandles;
//
//};