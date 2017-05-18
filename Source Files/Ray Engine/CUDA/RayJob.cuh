#pragma once

#include "Ray.cuh"
#include "Engine Core\Camera\Camera.h"


enum rayType {
	RayCOLOUR				//RayCOLOUR: A vec3 of RGB values to be displayed
	, RayCOLOUR_SPECTRAL	//RayCOLOUR_SPECTRAL: Uses additional processing time to perform spectral operations and returns the result in vec3 RGB space.
	, RayDISTANCE			//RayDISTANCE: A float of the specific ray's distance travelled.
	, RayOBJECT_ID			//RayOBJECT_ID: A unique ID of the first Object hit in a uint.
	, RayNORMAL				//RayNORMAL: The normal at the first point hit in a vec3.
	, RayUV					//RayUV: The UV at the first point hit in a vec2.
};

class RayJob
{
public:

	//@param The information to be retreiving from the job.
	//@param The number of rays/data-points to be cast into the scene.
	//@param The number of samples per ray or point that will be averaged into the result. Is more of a probability than number.
	//@param A camera that contains all the information to shoot a ray.
	//@param The amount of buffers used for result storage.
	__host__ RayJob(rayType, uint, float, Camera camera, void* resultsIn,int*);
	__host__ ~RayJob();

	//result variables

	void* results;
	int* groupData;

	//counting variables
	uint startIndex;

	//common variables
	Camera camera;
	rayType type;
	uint rayAmount;
	uint rayBaseAmount;
	float samples;
protected:

private:


};