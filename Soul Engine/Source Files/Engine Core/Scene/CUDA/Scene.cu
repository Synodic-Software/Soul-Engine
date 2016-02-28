#include "Scene.cuh"
#include "Algorithms\Data Algorithms\GPU Prefix Sum\PrefixSum.h"

Scene::Scene()
{
	objectsSize = 0;
	maxObjects = 100;
	indicesSize = 0;
	cudaMallocManaged(&objectList,
		maxObjects*sizeof(Object));
}


Scene::~Scene()
{
}
__host__ bool Scene::Clean(){
	/*bool* objectBitSetupTemp;


	int n = 0;
	for (int i = 0; i < objectsSize; i++){
		if (objectList[i].requestRemoval){



			n += objectList[i].faceAmount;
		}
	}*/

	return false;
}
__host__ bool Scene::Compile(){
	int n = 0;
	for (int i = 0; i < objectsSize;i++){
		if (!objectList[i].ready){
			n += objectList[i].faceAmount;
		}
	}

	if (n > 0){
		bool* objectBitSetupTemp;
		cudaMallocManaged(&objectBitSetupTemp, indicesSize + n*sizeof(bool));
		cudaMemcpy(objectBitSetupTemp, objectBitSetup, indicesSize, cudaMemcpyDefault);
		cudaFree(objectBitSetup);
		objectBitSetup = objectBitSetupTemp;
		int l = 0;
		for (int i = 0; i < objectsSize; i++){
			if (!objectList[i].ready){
				for (int t = 0; t < objectList[i].faceAmount;t++,l++){
					if (t==0){
						objectBitSetup[indicesSize + l] = true;
					}
					else{
						objectBitSetup[indicesSize + l] = false;
					}
				}
				objectList[i].ready = true;
			}
		}

	}

	return n > 0;
}

//__global__
__host__ void Scene::AttachObjIds(){
	PrefixSum::Calculate();
}
__host__ void Scene::Build(){
	bool a=Clean();
	bool b=Compile();


	//if neither clean or compile did anything
	if (a&&b){
		AttachObjIds();
	}
	


}

CUDA_FUNCTION glm::vec3 positionAlongRay(const Ray& ray, const float& t) {
	return ray.origin + t * ray.direction;
}
CUDA_FUNCTION glm::vec3 computeBackgroundColor(const glm::vec3& direction) {
	float position = (glm::dot(direction, normalize(glm::vec3(-0.5, 0.5, -1.0))) + 1) / 2.0f;
	return (1.0f - position) * glm::vec3(0.5f, 0.5f, 1.0f) + position * glm::vec3(1.0f, 1.0f, 1.0f);
}

CUDA_FUNCTION bool FindTriangleIntersect(const glm::vec3& a, const glm::vec3& b, const glm::vec3& c,
	const glm::vec3& o, const glm::vec3& d,
	float& t, float& bary1, float& bary2)
{
	glm::vec3 edge1 = b - a;
	glm::vec3 edge2 = c - a;

	glm::vec3 pvec = glm::cross(d, edge2);
	float det = glm::dot(edge1, pvec);
	if (det > -EPSILON && det < EPSILON){
		return false;
	}
	float inv_det = 1.0f / det;

	glm::vec3 tvec = o - a;
	bary1 = glm::dot(tvec, pvec) * inv_det;

	glm::vec3 qvec = glm::cross(tvec, edge1);
	bary2 = glm::dot(d, qvec) * inv_det;

	t = glm::dot(edge2, qvec) * inv_det;

	//bool hit = t>EPSILON&&(bary1 >= 0.0f && bary2 >= 0.0f && (bary1 + bary2) <= 1.0f);
	return t>EPSILON && (bary1 >= 0.0f && bary2 >= 0.0f && (bary1 + bary2) <= 1.0f);
}

CUDA_FUNCTION bool AABBIntersect(const glm::vec3& origin, const glm::vec3& extent, const glm::vec3& o, const glm::vec3& dInv, const float& t0, const float& t1){

	glm::vec3 boxMax = origin + extent;
	glm::vec3 boxMin = origin - extent;

	float tx1 = (boxMin.x - o.x)*dInv.x;
	float tx2 = (boxMax.x - o.x)*dInv.x;

	float tmin = glm::min(tx1, tx2);
	float tmax = glm::max(tx1, tx2);

	float ty1 = (boxMin.y - o.y)*dInv.y;
	float ty2 = (boxMax.y - o.y)*dInv.y;

	tmin = glm::max(tmin, glm::min(ty1, ty2));
	tmax = glm::min(tmax, glm::max(ty1, ty2));

	float tz1 = (boxMin.z - o.z)*dInv.z;
	float tz2 = (boxMax.z - o.z)*dInv.z;

	tmin = glm::max(tmin, glm::min(tz1, tz2));
	tmax = glm::min(tmax, glm::max(tz1, tz2));

	return tmax >= glm::max(t0, tmin) && tmin < t1;

}

CUDA_FUNCTION glm::vec3 Scene::IntersectColour(const Ray& ray)const{

	bool intersected=false;

	for (int o = 0; o < objectsSize;o++){

		Object* current = &objectList[o];

		for (int i = 0; i <current->faceAmount; i++){
			float bary1 = 0;
			float bary2 = 0;
			float lambda = 0;
			glm::uvec3 face = current->faces[i].indices;
			bool touched = FindTriangleIntersect(current->vertices[face.x].position, current->vertices[face.y].position, current->vertices[face.z].position,
				ray.origin, ray.direction,
				lambda, bary1, bary2);
			if (touched){
				intersected = true;
				break;
			}
		}
	}


	if (!intersected){
		return  computeBackgroundColor(ray.direction);
	}
	else{
		return glm::vec3(1.0f, 1.0f, 1.0f);
	}
}

__host__ void Scene::AddObject(Object& obj){
	if (maxObjects - 1 == objectsSize){
		std::cout << "ObjectMax reached" << std::endl;
		return;
	}
	objectList[objectsSize] = obj;
	objectsSize++;
}