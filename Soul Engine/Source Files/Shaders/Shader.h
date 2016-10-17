#pragma once

// Header Files
#include "Metrics.h"
#include "SoulCore.h"

// 3rd Party Libraries
#include "glm\glm.hpp"
#include "Utility\CUDA\CUDAHelper.cuh"
#include "Utility\CUDA\CUDADevices.cuh"

// Standard C++ Libraries
#include <exception>


/*

Class Shader:

Shader is an Abstract Base Class for all other shaders to inheret from.

This class specifies basic inputs and outputs for several functions which
are common across all shader implementations.



If you are implementing a new shader type, have it inherit from this class.

*/

/*

Note: Specifications for this class are still being developed.  Some methods may end up
having their permissions changed (i.e. public/private/protected)

*/





class Shader {
public:

	//Constructors and Destructors

	//Default Constructor

	/*
	This constructor takes no arguments

	Returns:
	A new shader object initialized with generic arguments.

	Note since this is an abstract base class it can not be instantiated
	directly.  This is instead intented as a template for drfault constructors
	in derived classes.

	Note: need for this constructor will be investigated in the future,
	at which point this constructor may be depricated
	*/
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

	Shader(glm::vec4 & storage, glm::vec4 & direction, glm::vec3 & origin, 
		glm::vec2 & hitCoord, const glm::vec3 & p1, const glm::vec3 & p2,
		const glm::vec3 & p3, const glm::vec3 & norm1, const glm::vec3 & norm2, const glm::vec3 & norm3,
		const glm::vec2 & tex1, const glm::vec2 & tex2, const glm::vec2 & tex3, Material* material);

	//Copy Constructor

	/*
	Shader constructor takes the following argument:
	
	const Shader & other: another shader, whose state will be copied in the new Shader

	Throws:
	std::invalid_argument if a non-compatiable Shader type is passed as other
	(Note: will have to further investigate project's exception handling methods
	to determine best way to handle this error and will likely revise specification)
	
	Returns:
	A new shader that is equivalent to the provided argument "other", provided
	that an exception is not thrown by the constructor

	Note: need for this constructor will be investigated in the future, 
	at which point this constructor may be depricated
	*/

	Shader(const Shader & other);
	
	//Destructor
	~Shader();




	//Member Functions



	//Accessors

	//Simple -- simple member field accessing




	//Derived -- quantities based on member fields

	//Returns whether or not a Face was hit by a Ray

	/*
	This method takes no arguments

	This method returns whether or not the face is hit by the ray.
	*/

	virtual bool isFaceHit() = 0;

	/*
	This method takes no arguments

	Returns:
	col vector of the shader.  This may vary depending on whether
	isFaceHit() is true or not.
	*/
	
	virtual glm::vec3 getCol() = 0;

	/*
	This method takes no arguments

	Returns:
	The best normal as defined by (norm2 - norm1) x (norm3 - norm1)
	where x represents the cross product.

	*/

	virtual glm::vec3 bestNormal() = 0;

	/*
	This method takes no arguments
	
	Returns:
	The best UV as defined by: 
	(1.0f - hitCoord.x - hitCoord.y) * tex1 + hitCoord.x * tex2 + hitCoord.y * tex3

	*/

	virtual glm::vec2 bestUV() = 0;

	/*
	This method takes no arguments

	Returns:
	Bias vector as defined by RAY_BIAS_DISTANCE * bestNormal() where
	RAY_BIAS_DISTANCE is a constant defined in the implementation

	*/

	virtual glm::vec3 biasVector() = 0;

	/*
	This method takes no arguments
	
	Returns:
	The best intersection point, which is the ray's origin + direction.w * origin

	*/

	virtual glm::vec3 bestIntersectionPoint() = 0;

	/*
	This medthod takes no arguments
	
	Returns:
	bestNormal() if bestNormal() . ray.direction_x,y,z < 0
	else -bestNormal() where
	. represents the dot product
	
	*/

	virtual glm::vec3 orientedNormal() = 0;

	/*
	This method takes no arguments
	
	Returns:
	the normalized version of the vector:

	(0,1,0) x orientedNormal() if |orientedNormal().x| > 0.1
	else (1,0,0) x orientedNormal()

	where x represents the cross product
	*/

	virtual glm::vec3 getU() = 0;


	/*
	This method takes no arguments
	
	Returns:
	orientedNormal() x getU() where
	x represents the cross product
	
	*/

	virtual glm::vec3 getV() = 0;





//Member Variables
protected:

	//Initialized at Object Creation
	glm::vec4 &storage, &direction;
	glm::vec3 &origin;
	glm::vec2 &hitCoord;
	const glm::vec3 &p1, &p2, &p3;
	const glm::vec3 &norm1, &norm2, &norm3;
	const glm::vec2 &tex1, &tex2, &tex3;
	Material *material;
};
