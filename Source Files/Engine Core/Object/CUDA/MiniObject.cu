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
	faceAmount = 1;
	tetAmount = 0;
	verticeAmount = 1;
	isStatic = false;
	requestRemoval = false;
}
