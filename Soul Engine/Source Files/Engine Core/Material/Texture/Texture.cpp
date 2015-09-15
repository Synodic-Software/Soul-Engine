#include "Texture.h"
#include <stdexcept>

static GLenum TextureFormatForBitmapFormat(Bitmap::Format format)
{
    switch (format) {
        case Bitmap::Format_Grayscale: return GL_LUMINANCE;
        case Bitmap::Format_GrayscaleAlpha: return GL_LUMINANCE_ALPHA;
        case Bitmap::Format_RGB: return GL_RGB;
        case Bitmap::Format_RGBA: return GL_RGBA;
        default: throw std::runtime_error("Unrecognised Bitmap::Format");
    }
}

Texture::Texture(const Bitmap& bitmap, GLint minMagFiler, GLint wrapMode) :
    _originalWidth((GLfloat)bitmap.width()),
    _originalHeight((GLfloat)bitmap.height())
{
	GLfloat fLargest;
	glGetFloatv (GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT,&fLargest);
	

    glGenTextures(1, &_object);
    glBindTexture(GL_TEXTURE_2D, _object);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY_EXT, fLargest);
    gluBuild2DMipmaps(GL_TEXTURE_2D,
                 TextureFormatForBitmapFormat(bitmap.format()),
                 (GLsizei)bitmap.width(), 
                 (GLsizei)bitmap.height(),
                 TextureFormatForBitmapFormat(bitmap.format()), 
                 GL_UNSIGNED_BYTE, 
                 bitmap.pixelBuffer());
    glBindTexture(GL_TEXTURE_2D, 0);
}

Texture::~Texture()
{
    glDeleteTextures(1, &_object);
}

GLuint Texture::object() const
{
    return _object;
}

GLfloat Texture::originalWidth() const
{
    return _originalWidth;
}

GLfloat Texture::originalHeight() const
{
    return _originalHeight;
}