#pragma once

#include "Engine Core/Object/Object.h"
#include "Photography/Camera/CUDA/Camera.cuh"
#include <vector>

/*
 *    Increment time.
 *    @param	parameter1	The first parameter.
 */

void IncrementTime(float);

/* A character. */
class Character : public Object{
public:

	/*
	 *    tag,distance,size.
	 *    @param	parameter1	The first parameter.
	 */

	virtual void Update(double)=0;
	/* Loads this object. */
	virtual void Load()=0;
	/* The camera */
	Camera camera;


	/* Information describing the callback */
	std::vector<bool*> CallbackInfo;
	/* Resets the callback information. */
	void ResetCallbackInfo();
private:
	
};