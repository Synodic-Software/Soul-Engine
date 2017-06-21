#pragma once

#include <string>

#include "OpenGLBackend.h"
#include "Raster Engine\Shader.h"

/* An open gl shader. */
/* An open gl shader. */
class OpenGLShader : public Shader {
public:

	/*
	 *    Constructor.
	 *
	 *    @param	parameter1	The first parameter.
	 *    @param	parameter2	The second parameter.
	 */

	OpenGLShader(std::string, shader_t);

	/*
	 *    Copy constructor.
	 *
	 *    @param	other	The other.
	 */

	OpenGLShader(const OpenGLShader& other);
	/* Destructor. */
	/* Destructor. */
	virtual ~OpenGLShader();

	/*
	 *    Gets the object.
	 *
	 *    @return	A GLuint.
	 */

	GLuint Object() const;

	/*
	 *    Assignment operator.
	 *
	 *    @param	other	The other.
	 *
	 *    @return	A shallow copy of this OpenGLShader.
	 */

	OpenGLShader& operator =(const OpenGLShader& other);

private:
	/* The object */
	/* The object */
	GLuint object;
};