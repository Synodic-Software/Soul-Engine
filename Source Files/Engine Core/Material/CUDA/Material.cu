#include "Material.cuh"
#include "Utility\CUDA\CUDAHelper.cuh"

Material::Material( std::string texName){



	CudaCheck(cudaDeviceSynchronize());
	image.LoadFromFile(texName.c_str(),false,true);
	CudaCheck(cudaDeviceSynchronize());

}
Material::~Material(){

}