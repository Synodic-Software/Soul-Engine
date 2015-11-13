#include "Scene.cuh"


Scene::Scene()
{
	objectsSize = 0;
	maxObjects = 100;

	cudaMallocManaged(&objectList,
		maxObjects*sizeof(Object));
}


Scene::~Scene()
{
}


CUDA_FUNCTION glm::vec3 positionAlongRay(const Ray& ray, const float& t) {
	return ray.origin + t * ray.direction;
}
CUDA_FUNCTION glm::vec3 computeBackgroundColor(const glm::vec3& direction) {
	float position = (dot(direction, normalize(glm::vec3(-0.5, 0.5, -1.0))) + 1) / 2;
	glm::vec3 interpolatedColor = (1.0f - position) * glm::vec3(0.5f, 0.5f, 1.0f) + position * glm::vec3(1.0f, 1.0f, 1.0f);
	return interpolatedColor * 1.0f;
}

CUDA_FUNCTION bool FindTriangleIntersect(const glm::vec3& a, const glm::vec3& b, const glm::vec3& c,
	const glm::vec3& o, const glm::vec3& d,
	float& lambda, float& bary1, float& bary2)
{
	glm::vec3 edge1 = b - a;
	glm::vec3 edge2 = c - a;

	glm::vec3 pvec = cross(d, edge2);
	float det = dot(edge1, pvec);
	if (det == 0.0f){
		return false;
	}
	float inv_det = 1.0f / det;

	glm::vec3 tvec = o - a;
	bary1 = dot(tvec, pvec) * inv_det;

	glm::vec3 qvec = cross(tvec, edge1);
	bary2 = dot(d, qvec) * inv_det;
	lambda = dot(edge2, qvec) * inv_det;

	bool hit = (bary1 >= 0.0f && bary2 >= 0.0f && (bary1 + bary2) <= 1.0f);
	return hit;
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

	for (uint i = 0; i < objectsSize;i++){

		Object* current = &objectList[i];

		for (uint i = 0; i <current->faceAmount; i++){
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
		glm::vec3 backColor = computeBackgroundColor(ray.direction);
		return glm::vec3(backColor.x, backColor.y, backColor.z);
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