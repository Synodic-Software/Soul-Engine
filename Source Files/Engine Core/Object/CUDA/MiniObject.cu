#include "MiniObject.cuh"
#include "Utility\CUDA\CUDAHelper.cuh"

MiniObject::MiniObject(Object& obj){

	verticeAmount = obj.verticeAmount;
	faceAmount = obj.faceAmount;
	tetAmount = obj.tetAmount;

	requestRemoval= obj.requestRemoval;
	isStatic = obj.isStatic;

}