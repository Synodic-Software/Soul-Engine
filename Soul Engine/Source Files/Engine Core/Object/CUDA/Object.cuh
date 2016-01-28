#pragma once

#include "Utility\CUDAIncludes.h"
#include "Utility/OpenGL/ShaderSupport.h"
#include "Utility/OpenGL/Shader.h"
#include "Engine Core/Material/Texture/Texture.h"
#include "Engine Core/Camera/CUDA/Camera.cuh"
#include "Input/Input.h"
#include "Engine Core/Material/Texture/Bitmap.h"
#include "Engine Core/Material/Material.h"
#include "Engine Core/Object/CUDA/Vertex.cuh"
#include "Engine Core/Object/CUDA/Face.cuh"
#include "Engine Core/Object/ObjLoader.h"

class Object: public Managed{
	public:

		Object();

		bool requestRemoval;
		bool isStatic;

		glm::vec3 xyzPosition;
		glm::vec3 velocity;
		glm::vec3 acceleration;

		/*	virtual void GetVertices(GLuint& buffer, GLuint& sizeVert);
			virtual void GetIndices(GLuint& buffer, GLuint& sizeIn);
			virtual void GetTextureCoords(GLuint& buffer, GLuint& sizeVert);
			virtual void GetNormals(GLuint& buffer, GLuint& sizeVert);
			virtual void GetMaterials(std::list<Material*>& material);
			virtual void GetPhysics(glm::vec3&, glm::vec3&,bool&);
			virtual void SetPhysics(glm::vec3&, glm::vec3&, bool&);*/

		void AddVertices(Vertex*,uint);
		void AddFaces(Face*, uint);
		void ExtractFromFile(const char*);

		uint verticeAmount;
		uint faceAmount;

		Vertex* vertices;
		Face* faces;

	virtual void Update(double) = 0;
	virtual void UpdateLate(double) = 0;
	virtual void Load() = 0;

	Material* materials;
	uint materialSize;

	uint localSceneIndex;
protected:
	
private:

	
};
