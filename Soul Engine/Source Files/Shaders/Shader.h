#pragma once

#include "Metrics.h"
#include "SoulCore.h"
#include "glm\glm.hpp"

class Shader {
public:

	//Constructors and Destructors

	//Default Constructor
	Shader();
	
	//Initialize Shader from argument provided data
	Shader(glm::vec4 & storage, glm::vec3 & direction, glm::vec3 & origin, 
		glm::vec2 & hitCoord, const glm::vec3 & p1, const glm::vec3 & p2,
		const glm::vec3 & p3, const glm::vec3 & norm1, const glm::vec3 & norm2, const glm::vec3 & norm3,
		const glm::vec2 & tex1, const glm::vec2 & tex2, const glm::vec2 & tex3, void* material);

	//Copy Constructor
	Shader(const Shader & other);
	
	//Destructor
	~Shader();

	//Member Functions

protected:
	glm::vec4 &storage;
	glm::vec3 &direction, &origin;
	glm::vec2 &hitCoord;
	const glm::vec3 &p1, &p2, &p3;
	const glm::vec3 &norm1, &norm2, &norm3;
	const glm::vec2 &tex1, &tex2, &tex3;
	void *material;
};

