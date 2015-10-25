#pragma once

#include "Engine Core/Object/Object.cuh"
#include "Engine Core/Camera/CUDA/Camera.cuh"
void IncrementTime(float);

class Character : public Object{
public:
	//tag,distance,size

	virtual void Update(double)=0;
	virtual void Load()=0;
	Camera camera;


	std::vector<bool*> CallbackInfo;
	void ResetCallbackInfo();
private:
	
};