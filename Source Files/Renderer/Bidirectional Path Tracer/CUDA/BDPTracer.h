//#pragma once
//
//#include "Engine Core/Object/Object.h"
//#include "Bounding Volume Heirarchy/BVH.h"
//#include "Renderer\Renderer.h"
//
//class BDPTracer: public Renderer{
//public:
//	BDPTracer( glm::vec2);
//	shading::ShaderSupport* shader;
//	shading::ShaderSupport* shaderCompute;
//	shading::ShaderSupport* shaderComputeI;
//	shading::ShaderSupport* shaderComputeColor;
//	GLuint vao;
//	GLuint vbo;
//	GLuint ibo;
//
//	bool continuous;
//	virtual void Render(Camera&, BVH*);
//	virtual void Draw();
//	virtual void Update(double dt);
//	void SwitchRender();
//	double startSwitch;
//	typedef struct
//	{
//		float XYZW[4];
//		float UV[2];
//		float Normal[3];
//	} Vertex;
//
//	GLuint iterationCounter;
//	GLuint traceCounter;
//	GLuint samples;
//	glm::vec2 screenSize;
//	GLuint Indices[6];
//	Vertex BlockData[4];
//	GLuint createTex();
//	GLuint display;
//	float apertureRadius;
//	float focalDistance;
//
//	typedef struct{
//		float x;
//		float y;
//		float z;
//		float w;
//	}Origin;
//
//	typedef struct{
//		float x;
//		float y;
//		float z;
//		float w;
//	}Direction;
//
//	typedef struct{
//		glm::vec4 absorption;
//		float reducedScattering;
//	}ASProperties;
//
//	typedef struct{
//		float refractive;
//		ASProperties asProperties;
//	}Medium;
//
//	typedef struct{
//		glm::vec4 diffuse;
//		glm::vec4 emitted;
//		glm::vec4 specular;
//		bool hasTransmission;
//		Medium medium;
//	}Material;
//
//	typedef struct{
//		glm::vec3 position;
//		glm::vec3 p0;
//		glm::vec3 p1;
//		glm::vec3 p2;
//		glm::vec3 n;
//		Material material;
//	}Poly;
//
//	typedef struct{
//		Origin origin;
//		Direction direction;
//	}Ray;
//
//	typedef struct{
//		glm::vec3 color;
//		float asProperty;
//	}asProp;
//
//	GLuint colorSSBO;
//	GLuint notAbsorpedSSBO;
//	GLuint raysDirSSBO;
//	GLuint raysOriSSBO;
//	GLuint propertiesSSBO;
//	int rayDepth;
//
//
//	glm::mat4 position;
//private:
//
//	std::string ResourcePath(std::string);
//	shading::ShaderSupport* LoadShaders(const char*, const char*, const char*, const char*, const char*);
//	shading::ShaderSupport* LoadShaders(const char*, const char*);
//	shading::ShaderSupport* LoadShaders(const char*);
//	shading::ShaderSupport* LoadShaders(const char*, const char*, const char*);
//
//
//	GLint counterUniform1;
//	GLint cameraApertureRadiusUniform;
//	GLint cameraFocalDistanceUniform;
//	GLint cameraPositionUniform;
//	GLint screenUniform;
//	GLint samplesUniform1;
//
//	GLint samplesUniform2;
//	GLint nodeSizeUniform;
//	GLint counterUniform2;
//
//	GLint continuousUniform;
//	GLint samplesUniform3;
//	GLint counterUniform3;
//
//	GLint texUniform;
//	GLint cameraUniform;
//	GLint modelUniform;
//
//	GLint horizontalAxisUniform;
//	GLint verticalAxisUniform;
//	GLint middleUniform;
//	GLint horizontalUniform;
//	GLint verticalUniform;
//};
