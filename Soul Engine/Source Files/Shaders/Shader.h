#pragma once

#include "Metrics.h"
#include "SoulCore.h"
#include "glm\glm.hpp"


/*

Class Shader:

Shader is an Abstract Base Class for all other shaders to inheret from.

This class specifies basic inputs and outputs for several functions which
are common across all shader implementations.



If you are implementing a new shader type, have it inherit from this class.

*/





class Shader {
public:

	//Constructors and Destructors

	//Default Constructor
	Shader();
	
	//Initialize Shader from argument provided data

	/*

	Shader Constructor takes the following arguments:

	glm::vec4 storage: the energy of the light a ray is carrying

	glm::vec3 direction: the direction the ray is traveling in

	glm::vec3 origin the origin of the ray

	void * material: holds arbitrary material info
	(Note that void* is just a temporary type holder
	and will eventually be updated to a proper material 
	class's pointer)

	glm::vec2 hitCoord: barycentric hit coordinate

	
	The following arguments all deal with a face to be hit:

	glm::vec3 pos1, pos2, pos2: the positions of the face

	glm::vec3 norm1, norm2, norm3: the normals of the face

	glm::vec3 tex1, tex2, tex3: the texture coordinates of the face
	
	Returns:
	A new Shader object initialized with the above values
	*/

	Shader(glm::vec4 & storage, glm::vec3 & direction, glm::vec3 & origin, 
		glm::vec2 & hitCoord, const glm::vec3 & p1, const glm::vec3 & p2,
		const glm::vec3 & p3, const glm::vec3 & norm1, const glm::vec3 & norm2, const glm::vec3 & norm3,
		const glm::vec2 & tex1, const glm::vec2 & tex2, const glm::vec2 & tex3, void* material);

	//Copy Constructor
	Shader(const Shader & other);
	
	//Destructor
	~Shader();




	//Member Functions

	//Returns whether or not a Face was hit by a Ray
	virtual void isFaceHit() = 0;




//Member Variables
protected:

	//Initialized at Object Creation
	glm::vec4 &storage;
	glm::vec3 &direction, &origin;
	glm::vec2 &hitCoord;
	const glm::vec3 &p1, &p2, &p3;
	const glm::vec3 &norm1, &norm2, &norm3;
	const glm::vec2 &tex1, &tex2, &tex3;
	void *material;
};
