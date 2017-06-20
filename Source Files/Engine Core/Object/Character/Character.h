//---------------------------------------------------------------------------------------------------
//@file	N:\Documents\Soul Engine\Source Files\Engine Core\Object\Character\Character.h.
//Declares the character class.

#pragma once

#include "Engine Core/Object/Object.h"
#include "Engine Core/Camera/CUDA/Camera.cuh"
#include <vector>

//---------------------------------------------------------------------------------------------------
//Increment time.
//@param	parameter1	The first parameter.

void IncrementTime(float);

//A character.
class Character : public Object{
public:

	//---------------------------------------------------------------------------------------------------
	//tag,distance,size.
	//@param	parameter1	The first parameter.

	virtual void Update(double)=0;
	//Loads this object.
	virtual void Load()=0;
	//The camera
	Camera camera;


	//Information describing the callback
	std::vector<bool*> CallbackInfo;
	//Resets the callback information.
	void ResetCallbackInfo();
private:
	
};