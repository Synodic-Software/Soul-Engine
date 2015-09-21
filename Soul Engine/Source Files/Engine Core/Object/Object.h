#pragma once

#include "Engine Core/BasicDependencies.h"
#include "Utility/OpenGL/ShaderSupport.h"
#include "Utility/OpenGL/Shader.h"
#include "Engine Core/Material/Texture/Texture.h"
#include "Engine Core/Camera/Camera.h"
#include "Input/Input.h"
#include "Engine Core/Material/Texture/Bitmap.h"
#include "Engine Core/Material/Material.h"
#include "ObjectProperties.h"


class Object{
	public:
		Object();
		ObjectProperties* properties;

		bool requestRemoval;

		std::atomic<bool> draw;
		std::atomic<bool> initialized;
		std::atomic<bool> loaded;
		bool hasPhysics;


		boolean needLoad;

			bool isStatic;
			glm::vec3 xyzPosition;
			glm::vec3 velocity;
			glm::vec3 acceleration;

			/*Texture* texture;*/

		/*	virtual void GetVertices(GLuint& buffer, GLuint& sizeVert);
			virtual void GetIndices(GLuint& buffer, GLuint& sizeIn);
			virtual void GetTextureCoords(GLuint& buffer, GLuint& sizeVert);
			virtual void GetNormals(GLuint& buffer, GLuint& sizeVert);
			virtual void GetMaterials(std::list<Material*>& material);
			virtual void GetPhysics(glm::vec3&, glm::vec3&,bool&);
			virtual void SetPhysics(glm::vec3&, glm::vec3&, bool&);*/

			virtual void AddVertices(std::vector<Vertex>&);
			virtual void AddFaces(std::vector<Face>&);

			GLuint verticeAmount;
			GLuint indiceAmount;


	virtual void Update(double) = 0;
	virtual void UpdateLate(double) = 0;
	virtual void Load() = 0;
	//std::vector<instance> instances;

	std::list<Material> materials;

protected:
	
	//typedef OpenMesh::PolyMesh_ArrayKernelT<>  Mesh;
	//	Mesh mesh;

	//	Object::Mesh::VertexHandle Object::AddVertex(float x, float y, float z);
	//	Object::Mesh::VertexHandle Object::AddVertex(glm::vec3);
	//	Object::Mesh::VertexHandle Object::AddVertex(glm::vec4);

	//	void AddFace(Object::Mesh::VertexHandle, Object::Mesh::VertexHandle, Object::Mesh::VertexHandle);

		//std::string ResourcePath(std::string);
		//shading::ShaderSupport* LoadShaders(const char*, const char*, const char*, const char*, const char*);
		//shading::ShaderSupport* LoadShaders(const char*, const char*);
		//shading::ShaderSupport* LoadShaders(const char*);
		//shading::ShaderSupport* LoadShaders(const char*, const char*, const char*);
		private:

//	int GetTag();
//	virtual void Update(double, float, glm::vec3) = 0;
//	virtual void Initialize() = 0;
//	virtual void InitializeVO() = 0;
//	virtual void Load() = 0;
//	shading::ShaderSupport* shader;
//	Texture* texture;
//	glm::vec3 sunNormal;


	
};
