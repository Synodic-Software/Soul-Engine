#include "Material.cuh"

Bitmap LoadBmp(const char* filename) {
	Bitmap bmp = Bitmap::bitmapFromFile(filename);
	bmp.flipVertically();
	return bmp;
}


#define MAX(a,b) ((a > b) ? a : b)

uint getMipMapLevels(cudaExtent size)
{
	size_t sz = MAX(MAX(size.width, size.height), size.depth);

	uint levels = 0;

	while (sz)
	{
		sz /= 2;
		levels++;
	}

	return levels;
}
inline void _checkHost(bool test, const char *condition, const char *file, int line, const char *func)
{
	if (!test)
	{
		fprintf(stderr, "HOST error at %s:%d (%s) \"%s\" \n",
			file, line, condition, func);
		exit(EXIT_FAILURE);
	}
}

#define checkHost(condition)   _checkHost(condition, #condition,__FILE__,__LINE__,__FUNCTION__)






__device__ __inline__ uchar4 to_uchar4(float4 vec)
{
	return make_uchar4((unsigned char)vec.x, (unsigned char)vec.y, (unsigned char)vec.z, (unsigned char)vec.w);
}




__global__ void
d_mipmap(cudaSurfaceObject_t mipOutput, cudaTextureObject_t mipInput, uint imageW, uint imageH)
{
	uint x = blockIdx.x * blockDim.x + threadIdx.x;
	uint y = blockIdx.y * blockDim.y + threadIdx.y;

	float px = 1.0 / float(imageW);
	float py = 1.0 / float(imageH);


	if ((x < imageW) && (y < imageH))
	{
		// take the average of 4 samples

		// we are using the normalized access to make sure non-power-of-two textures
		// behave well when downsized.
		float4 color =
			(tex2D<float4>(mipInput, (x + 0) * px, (y + 0) * py)) +
			(tex2D<float4>(mipInput, (x + 1) * px, (y + 0) * py)) +
			(tex2D<float4>(mipInput, (x + 1) * px, (y + 1) * py)) +
			(tex2D<float4>(mipInput, (x + 0) * px, (y + 1) * py));


		color.x /= 4.0;
		color.y /= 4.0;
		color.z /= 4.0;
		color.w /= 4.0;
		color.x *= 255.0;
		color.y *= 255.0;
		color.z *= 255.0;
		color.w *= 255.0;
		color = fminf(color, make_float4(255.0, 255.0, 255.0, 255.0));

		surf2Dwrite(to_uchar4(color), mipOutput, x * sizeof(uchar4), y);
	}
}



void generateMipMaps(cudaMipmappedArray_t mipmapArray, cudaExtent size)
{
	size_t width = size.width;
	size_t height = size.height;


	uint level = 0;

	while (width != 1 || height != 1)
	{
		width /= 2;
		width = MAX((size_t)1, width);
		height /= 2;
		height = MAX((size_t)1, height);

		cudaArray_t levelFrom;
		CudaCheck(cudaGetMipmappedArrayLevel(&levelFrom, mipmapArray, level));
		cudaArray_t levelTo;
		CudaCheck(cudaGetMipmappedArrayLevel(&levelTo, mipmapArray, level + 1));

		cudaExtent  levelToSize;
		CudaCheck(cudaArrayGetInfo(NULL, &levelToSize, NULL, levelTo));
		checkHost(levelToSize.width == width);
		checkHost(levelToSize.height == height);
		checkHost(levelToSize.depth == 0);

		// generate texture object for reading
		cudaTextureObject_t         texInput;
		cudaResourceDesc            texRes;
		memset(&texRes, 0, sizeof(cudaResourceDesc));

		texRes.resType = cudaResourceTypeArray;
		texRes.res.array.array = levelFrom;

		cudaTextureDesc             texDescr;
		memset(&texDescr, 0, sizeof(cudaTextureDesc));

		texDescr.normalizedCoords = 1;
		texDescr.filterMode = cudaFilterModeLinear;

		texDescr.addressMode[0] = cudaAddressModeClamp;
		texDescr.addressMode[1] = cudaAddressModeClamp;
		texDescr.addressMode[2] = cudaAddressModeClamp;

		texDescr.readMode = cudaReadModeNormalizedFloat;

		CudaCheck(cudaCreateTextureObject(&texInput, &texRes, &texDescr, NULL));

		// generate surface object for writing

		cudaSurfaceObject_t surfOutput;
		cudaResourceDesc    surfRes;
		memset(&surfRes, 0, sizeof(cudaResourceDesc));
		surfRes.resType = cudaResourceTypeArray;
		surfRes.res.array.array = levelTo;

		CudaCheck(cudaCreateSurfaceObject(&surfOutput, &surfRes));

		// run mipmap kernel
		dim3 blockSize(16, 16, 1);
		dim3 gridSize(((uint)width + blockSize.x - 1) / blockSize.x, ((uint)height + blockSize.y - 1) / blockSize.y, 1);

		d_mipmap << <gridSize, blockSize >> >(surfOutput, texInput, (uint)width, (uint)height);

		CudaCheck(cudaDeviceSynchronize());
		CudaCheck(cudaGetLastError());

		CudaCheck(cudaDestroySurfaceObject(surfOutput));

		CudaCheck(cudaDestroyTextureObject(texInput));

		level++;
	}
}


Material::Material( std::string texName){

	//cudaTextureObject_t texObj;
	//cudaArray* cuArray;
	//Bitmap bmp;

	//CudaCheck(cudaDeviceSynchronize());
	//bmp = LoadBmp(texName.c_str());


	//Image image;
	//image.size = make_cudaExtent(bmp.width(), bmp.height(), 0);
	//image.h_data = bmp.pixelBuffer();


	//uint levels = getMipMapLevels(image.size);

	//////Allocate CUDA array in device memory 
	//std::cout << bmp.height() << " " << bmp.width() << " " << sizeof(float) << std::endl;

	////for (int i = 0; i < bmp.width(); i++){
	////	for (int c = 0; c < bmp.height(); c++){
	////		std::cout << (short)bmp.pixelBuffer()[c] << " ";
	////	}
	////	std::cout << std::endl;
	////}


	//cudaChannelFormatDesc desc = cudaCreateChannelDesc<uchar4>();
	//CudaCheck(cudaMallocMipmappedArray(&image.mipmapArray, &desc, image.size, levels));

	//cudaArray_t level0;
	//CudaCheck(cudaGetMipmappedArrayLevel(&level0, image.mipmapArray, 0));

	//cudaMemcpy3DParms copyParams = { 0 };
	//copyParams.srcPtr = make_cudaPitchedPtr(image.h_data, image.size.width * sizeof(uchar4), image.size.width, image.size.height);
	//copyParams.dstArray = level0;
	//copyParams.extent = image.size;
	//copyParams.extent.depth = 1;
	//copyParams.kind = cudaMemcpyHostToDevice;
	//CudaCheck(cudaMemcpy3D(&copyParams));

	//// compute rest of mipmaps based on level 0
	//generateMipMaps(image.mipmapArray, image.size);

	//// generate bindless texture object

	//cudaResourceDesc            resDescr;
	//memset(&resDescr, 0, sizeof(cudaResourceDesc));

	//resDescr.resType = cudaResourceTypeMipmappedArray;
	//resDescr.res.mipmap.mipmap = image.mipmapArray;

	//cudaTextureDesc             texDescr;
	//memset(&texDescr, 0, sizeof(cudaTextureDesc));

	//texDescr.normalizedCoords = 1;
	//texDescr.filterMode = cudaFilterModeLinear;
	//texDescr.mipmapFilterMode = cudaFilterModeLinear;

	//texDescr.addressMode[0] = cudaAddressModeWrap;
	//texDescr.addressMode[1] = cudaAddressModeWrap;
	//texDescr.addressMode[2] = cudaAddressModeWrap;

	//texDescr.maxMipmapLevelClamp = float(levels - 1);

	//texDescr.readMode = cudaReadModeNormalizedFloat;

	//CudaCheck(cudaCreateTextureObject(&texObj, &resDescr, &texDescr, NULL));

	//CudaCheck(cudaDeviceSynchronize());
}
Material::~Material(){

	// Destroy texture object 
	//cudaDestroyTextureObject(texObj); 

	//// Free device memory 
	//cudaFree(cuArray);

	//make a cleanup function

	//if (texture != NULL){
	//	delete texture;
	//}
	//if (defaultTexture!=NULL){
	//	delete defaultTexture;
	//}
}
//void Material::SetTexture(std::string name ){
//	//Bitmap bmp = Bitmap::bitmapFromFile(name);
//		//bmp.flipVertically();
//		//texture=new Texture(bmp);
//		//textureHandle = glGetTextureHandleARB(texture->object());
//	//textureIsLoaded = true;
//}
//void Material::SetDefaultTexture(std::string name){
//	//Bitmap bmp = Bitmap::bitmapFromFile(name);
//	//bmp.flipVertically(); 
//	//defaultTexture = new Texture(bmp);
//}