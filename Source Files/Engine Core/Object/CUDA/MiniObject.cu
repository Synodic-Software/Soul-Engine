#include "MiniObject.cuh"
#include "Utility\CUDA\CUDAHelper.cuh"

MiniObject::MiniObject(Object& obj){

	verticeAmount = obj.verticeAmount;
	faceAmount = obj.faceAmount;
	tetAmount = obj.tetAmount;

	requestRemoval= obj.requestRemoval;
	isStatic = obj.isStatic;

}

MiniObject::MiniObject() {

	verticeAmount = 0;
	faceAmount = 0;
	tetAmount = 0;

	requestRemoval = false;
	isStatic = false;

}