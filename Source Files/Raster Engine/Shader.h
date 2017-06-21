#pragma once

#include <string>
#include "Utility\Includes\GLFWIncludes.h"

/* Values that represent shader ts. */
/* Values that represent shader ts. */
enum shader_t {VERTEX_SHADER,FRAGMENT_SHADER};

/* A shader. */
/* A shader. */
class Shader {
public:

	/*
	 *    Constructor.
	 *
	 *    @param	parameter1	The first parameter.
	 *    @param	parameter2	The second parameter.
	 */

	Shader(std::string, shader_t);
	/* Destructor. */
	/* Destructor. */
	virtual ~Shader();

	/*
	 *    Extracts the shader described by parameter1.
	 *
	 *    @param	parameter1	The first parameter.
	 */

	void ExtractShader(const std::string&);

protected:

	/* The name */
	/* The name */
	std::string name;
	/* The code string */
	/* The code string */
	std::string codeStr;
	/* The type */
	/* The type */
	shader_t type;
	/* Number of references */
	/* Number of references */
	unsigned* referenceCount;
	/* Retains this Shader. */
	/* Retains this Shader. */
	void Retain();
	/* Releases this Shader. */
	/* Releases this Shader. */
	void Release();

private:
	

};