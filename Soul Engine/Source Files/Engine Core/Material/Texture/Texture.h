#pragma once

#include <GL/glew.h>
#include "Bitmap.h"
#include "Engine Core\BasicDependencies.h"
    
#pragma pack(push,4)
struct Image
{
	void                   *h_data;
	cudaExtent              size;
	cudaResourceType        type;
	cudaArray_t             dataArray;
	cudaMipmappedArray_t    mipmapArray;
	cudaTextureObject_t     textureObject;

	Image()
	{
		memset(this, 0, sizeof(Image));
	}
};
#pragma pack(pop)


    /**
     Represents an OpenGL texture
     */
    class Texture {
    public:
        /**
         Creates a texture from a bitmap.
         
         The texture will be loaded upside down because shading::Bitmap pixel data
         is ordered from the top row down, but OpenGL expects the data to
         be from the bottom row up.
         
         @param bitmap  The bitmap to load the texture from
         @param minMagFiler  GL_NEAREST or GL_LINEAR
         @param wrapMode GL_REPEAT, GL_MIRRORED_REPEAT, GL_CLAMP_TO_EDGE, or GL_CLAMP_TO_BORDER
         */
        Texture(const Bitmap& bitmap,
                GLint minMagFiler = GL_LINEAR,
                GLint wrapMode = GL_CLAMP_TO_EDGE);
        
        /**
         Deletes the texture object with glDeleteTextures
         */
        ~Texture();
        
        /**
         @result The texure object, as created by glGenTextures
         */
        GLuint object() const;
        
        /**
         @result The original width (in pixels) of the bitmap this texture was made from
         */
        GLfloat originalWidth() const;

        /**
         @result The original height (in pixels) of the bitmap this texture was made from
         */
        GLfloat originalHeight() const;
        
    private:
        GLuint _object;
        GLfloat _originalWidth;
        GLfloat _originalHeight;
        
        //copying disabled
        Texture(const Texture&);
        const Texture& operator=(const Texture&);
    };
  
