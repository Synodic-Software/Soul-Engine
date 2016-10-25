
#include "ShaderSupport.h"
#include <stdexcept>
#include <iostream>
#include <glm/gtc/type_ptr.hpp>

using namespace shading;

ShaderSupport::ShaderSupport(const std::vector<Shader>& shaders) :
    _object(0)
{
    if(shaders.size() <= 0)
        throw std::runtime_error("No shaders were provided to create the program");
    
    //create the program object
    _object = glCreateProgram();
    if(_object == 0)
        throw std::runtime_error("glCreateProgram failed");
    
    //attach all the shaders
    for(unsigned i = 0; i < shaders.size(); ++i)
        glAttachShader(_object, shaders[i].object());

    //link the shaders together
    glLinkProgram(_object);
    
    //detach all the shaders
    for(unsigned i = 0; i < shaders.size(); ++i)
        glDetachShader(_object, shaders[i].object());
    
    //throw exception if linking failed
    GLint status;
    glGetProgramiv(_object, GL_LINK_STATUS, &status);
    if (status == GL_FALSE) {

//			std::string msg("Program linking failure in " + shaders[0].name + ": ");


		std::string msg("Program linking failure in: ");
        GLint infoLogLength;
        glGetProgramiv(_object, GL_INFO_LOG_LENGTH, &infoLogLength);
        char* strInfoLog = new char[infoLogLength + 1];
        glGetProgramInfoLog(_object, infoLogLength, NULL, strInfoLog);
        msg += strInfoLog;
        delete[] strInfoLog;
        
        glDeleteProgram(_object); _object = 0;
		std::cerr << msg << std::endl;
        throw std::runtime_error(msg);
    }
}

ShaderSupport::~ShaderSupport() {
    //might be 0 if ctor fails by throwing exception
    if(_object != 0) glDeleteProgram(_object);
}

GLuint ShaderSupport::object() const {
    return _object;
}

void ShaderSupport::use() const {
    glUseProgram(_object);
}



void ShaderSupport::stopUsing() const {
    glUseProgram(0);
}

GLint ShaderSupport::attrib(const GLchar* attribName) const {
    if(!attribName)
        throw std::runtime_error("attribName was NULL");
    
    GLint attrib = glGetAttribLocation(_object, attribName);
    if(attrib == -1)
        throw std::runtime_error(std::string("Program attribute not found: ") + attribName);
    
    return attrib;
}

GLint ShaderSupport::uniform(const GLchar* uniformName) const {
    if(!uniformName)
        throw std::runtime_error("uniformName was NULL");
    
    GLint uniform = glGetUniformLocation(_object, uniformName);
	if (uniform == -1){
		std::cout << std::string("Program uniform not found: ") + uniformName << std::endl;
		throw std::runtime_error(std::string("Program uniform not found: ") + uniformName);
	}
    
    return uniform;
}

#define ATTRIB_N_UNIFORM_SETTERS(OGL_TYPE, TYPE_PREFIX, TYPE_SUFFIX) \
\
    void ShaderSupport::setAttrib(const GLint pos, OGL_TYPE v0) \
        {  glVertexAttrib ## TYPE_PREFIX ## 1 ## TYPE_SUFFIX (pos, v0); } \
    void ShaderSupport::setAttrib(const GLint pos, OGL_TYPE v0, OGL_TYPE v1) \
        {  glVertexAttrib ## TYPE_PREFIX ## 2 ## TYPE_SUFFIX (pos, v0, v1); } \
    void ShaderSupport::setAttrib(const GLint pos, OGL_TYPE v0, OGL_TYPE v1, OGL_TYPE v2) \
        {  glVertexAttrib ## TYPE_PREFIX ## 3 ## TYPE_SUFFIX (pos, v0, v1, v2); } \
    void ShaderSupport::setAttrib(const GLint pos, OGL_TYPE v0, OGL_TYPE v1, OGL_TYPE v2, OGL_TYPE v3) \
        {  glVertexAttrib ## TYPE_PREFIX ## 4 ## TYPE_SUFFIX (pos, v0, v1, v2, v3); } \
\
    void ShaderSupport::setAttrib1v(const GLint pos, const OGL_TYPE* v) \
        {  glVertexAttrib ## TYPE_PREFIX ## 1 ## TYPE_SUFFIX ## v (pos, v); } \
    void ShaderSupport::setAttrib2v(const GLint pos, const OGL_TYPE* v) \
        {  glVertexAttrib ## TYPE_PREFIX ## 2 ## TYPE_SUFFIX ## v (pos, v); } \
    void ShaderSupport::setAttrib3v(const GLint pos, const OGL_TYPE* v) \
        {  glVertexAttrib ## TYPE_PREFIX ## 3 ## TYPE_SUFFIX ## v (pos, v); } \
    void ShaderSupport::setAttrib4v(const GLint pos, const OGL_TYPE* v) \
        {  glVertexAttrib ## TYPE_PREFIX ## 4 ## TYPE_SUFFIX ## v (pos, v); } \
\
    void ShaderSupport::setUniform(const GLint pos, OGL_TYPE v0) \
        {  glUniform1 ## TYPE_SUFFIX (pos, v0); } \
    void ShaderSupport::setUniform(const GLint pos, OGL_TYPE v0, OGL_TYPE v1) \
        {  glUniform2 ## TYPE_SUFFIX (pos, v0, v1); } \
    void ShaderSupport::setUniform(const GLint pos, OGL_TYPE v0, OGL_TYPE v1, OGL_TYPE v2) \
        {  glUniform3 ## TYPE_SUFFIX (pos, v0, v1, v2); } \
    void ShaderSupport::setUniform(const GLint pos, OGL_TYPE v0, OGL_TYPE v1, OGL_TYPE v2, OGL_TYPE v3) \
        {  glUniform4 ## TYPE_SUFFIX (pos, v0, v1, v2, v3); } \
\
    void ShaderSupport::setUniform1v(const GLint pos, const OGL_TYPE* v, GLsizei count) \
        {  glUniform1 ## TYPE_SUFFIX ## v (pos, count, v); } \
    void ShaderSupport::setUniform2v(const GLint pos, const OGL_TYPE* v, GLsizei count) \
        {  glUniform2 ## TYPE_SUFFIX ## v (pos, count, v); } \
    void ShaderSupport::setUniform3v(const GLint pos, const OGL_TYPE* v, GLsizei count) \
        {  glUniform3 ## TYPE_SUFFIX ## v (pos, count, v); } \
    void ShaderSupport::setUniform4v(const GLint pos, const OGL_TYPE* v, GLsizei count) \
        {  glUniform4 ## TYPE_SUFFIX ## v (pos, count, v); }

ATTRIB_N_UNIFORM_SETTERS(GLfloat, , f);
ATTRIB_N_UNIFORM_SETTERS(GLdouble, , d);
ATTRIB_N_UNIFORM_SETTERS(GLint, I, i);
ATTRIB_N_UNIFORM_SETTERS(GLuint, I, ui);

void ShaderSupport::setUniformMatrix2(const GLint pos, const GLfloat* v, GLsizei count, GLboolean transpose) {
	glUniformMatrix2fv(pos, count, transpose, v);
}

void ShaderSupport::setUniformMatrix3(const GLint pos, const GLfloat* v, GLsizei count, GLboolean transpose) {
	glUniformMatrix3fv(pos, count, transpose, v);
}

void ShaderSupport::setUniformMatrix4(const GLint pos, const GLfloat* v, GLsizei count, GLboolean transpose) {
	glUniformMatrix4fv(pos, count, transpose, v);
}

void ShaderSupport::setUniform(const GLint pos, const glm::mat2& m, GLboolean transpose) {
	glUniformMatrix2fv(pos, 1, transpose, glm::value_ptr(m));
}

void ShaderSupport::setUniform(const GLint pos, const glm::mat3& m, GLboolean transpose) {
	glUniformMatrix3fv(pos, 1, transpose, glm::value_ptr(m));
}

void ShaderSupport::setUniform(const GLint pos, const glm::mat4& m, GLboolean transpose) {
	glUniformMatrix4fv(pos, 1, transpose, glm::value_ptr(m));
}

shading::ShaderSupport*  LoadShaders(const char* vertFilename, const char* controlFilename, const char* evaluationFilename, const char* geometryFilename, const char* fragFilename) {
	std::vector<shading::Shader> shaders;
	shaders.push_back(shading::Shader::shaderFromFile(vertFilename, GL_VERTEX_SHADER));
	shaders.push_back(shading::Shader::shaderFromFile(controlFilename, GL_TESS_CONTROL_SHADER));
	shaders.push_back(shading::Shader::shaderFromFile(evaluationFilename, GL_TESS_EVALUATION_SHADER));
	shaders.push_back(shading::Shader::shaderFromFile(geometryFilename, GL_GEOMETRY_SHADER));
	shaders.push_back(shading::Shader::shaderFromFile(fragFilename, GL_FRAGMENT_SHADER));
	return new shading::ShaderSupport(shaders);
}

shading::ShaderSupport*  LoadShaders(const char* vertFilename, const char* fragFilename) {
	std::vector<shading::Shader> shaders;
	shaders.push_back(shading::Shader::shaderFromFile(vertFilename, GL_VERTEX_SHADER));
	shaders.push_back(shading::Shader::shaderFromFile(fragFilename, GL_FRAGMENT_SHADER));
	return new shading::ShaderSupport(shaders);
}
shading::ShaderSupport*  LoadShaders(const char* compFilename) {
	std::vector<shading::Shader> shaders;
	shaders.push_back(shading::Shader::shaderFromFile(compFilename, GL_COMPUTE_SHADER));
	return new shading::ShaderSupport(shaders);
}
shading::ShaderSupport*  LoadShaders(const char* vertFilename, const char* geometryFilename, const char* fragFilename) {
	std::vector<shading::Shader> shaders;
	shaders.push_back(shading::Shader::shaderFromFile(vertFilename, GL_VERTEX_SHADER));
	shaders.push_back(shading::Shader::shaderFromFile(geometryFilename, GL_GEOMETRY_SHADER));
	shaders.push_back(shading::Shader::shaderFromFile(fragFilename, GL_FRAGMENT_SHADER));
	return new shading::ShaderSupport(shaders);
}