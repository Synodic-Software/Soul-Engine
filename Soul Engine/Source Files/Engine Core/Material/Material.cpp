#include "Material.h"

Material::Material(){
	hasTexture = true;
	//texture = defaultTexture;
	textureIsLoaded = false;
}
Material::~Material(){

	//make a cleanup function

	//if (texture != NULL){
	//	delete texture;
	//}
	//if (defaultTexture!=NULL){
	//	delete defaultTexture;
	//}
}
void Material::SetTexture(std::string name ){
	//Bitmap bmp = Bitmap::bitmapFromFile(name);
		//bmp.flipVertically();
		//texture=new Texture(bmp);
		//textureHandle = glGetTextureHandleARB(texture->object());
	//textureIsLoaded = true;
}
void Material::SetDefaultTexture(std::string name){
	//Bitmap bmp = Bitmap::bitmapFromFile(name);
	//bmp.flipVertically(); 
	//defaultTexture = new Texture(bmp);
}