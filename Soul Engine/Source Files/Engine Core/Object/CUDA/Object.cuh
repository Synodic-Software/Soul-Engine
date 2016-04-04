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
#include "Bounding Volume Heirarchy\BoundingBox.h"

class Face;

class Object: public Managed{
	public:

		Object();

		Object(glm::vec3,std::string,Material*);

		bool requestRemoval;
		bool ready;
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

	void Update(double);
	void UpdateLate(double);
	void Load();

	Material** materialP;
	uint materialSize;

	BoundingBox box;

	uint localSceneIndex;
protected:
	
private:

	
};
