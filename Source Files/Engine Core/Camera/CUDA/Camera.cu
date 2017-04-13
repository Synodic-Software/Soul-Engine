#include "Engine Core/Camera/CUDA/Camera.cuh"
#include <glm/gtx/rotate_vector.hpp>

__host__ __device__ Camera::Camera() :
	resolution(0,0),
	aspectRatio(0),
    position(0.0f,0.0f,0.0f),
	forward(0.0f,0.0f,1.0f),
	right(1.0f, 0.0f, 0.0f),
	fieldOfView(90.0f,65.0f),
	aperture(2*MILLIMETER),
	focalDistance(17*MILLIMETER),
	circularDistribution(false)
{	
}

__host__ __device__  Camera::~Camera(){

}

__host__ __device__ void Camera::SetAspect(float newA){
	aspectRatio = newA;
}
__host__ __device__ float Camera::GetAspect(){
	return aspectRatio;
}

__host__ __device__ glm::vec3 Camera::Position() const {
    return position;
}

__host__ __device__ void Camera::SetPosition(const glm::vec3& positionN) {
    position = positionN;
}

__host__ __device__ void Camera::OffsetPosition(const glm::vec3& offset) {
    position += offset;
}

__host__ __device__ glm::vec2 Camera::FieldOfView() const{
    return fieldOfView;
}
__host__ __device__ void Camera::SetFieldOfView(glm::vec2 fieldOfViewIn) {
    fieldOfView = fieldOfViewIn;
}



__host__ __device__ glm::vec3 Camera::Forward() const {
    return forward;
}
__host__ __device__ void Camera::SetForward(glm::vec3& forN){
	forward = glm::normalize(forN);
}

__host__ __device__ glm::vec3 Camera::Right() const {
    return right;
}
__host__ __device__ void Camera::SetRight(glm::vec3& rightn) {
	right = normalize(rightn);
}


__device__ void Camera::SetupRay(const uint& index, Ray& ray, curandState& randState){

	uint y = index / resolution.x;

	float sx = ((curand_uniform(&randState) - 0.5f) + (index %resolution.x)) / (resolution.x - 1);
	float sy = ((curand_uniform(&randState) - 0.5f) + y) / (resolution.y - 1);

	float angle = TWO_PI * curand_uniform(&randState);
	float distance = aperture * sqrt(curand_uniform(&randState));


	/*ALTERNATE aperaturPoint
	+ ((cos(angle) * distance) * right) + ((sin(angle) * distance) * verticalAxis)*/

	glm::vec3 aperturePoint = position ;
	
	glm::vec3 pointOnPlaneOneUnitAwayFromEye = 
		(aperturePoint + forward) + (((2 * sx) - 1) * xHelper) + (((2 * sy) - 1) * yHelper);

	ray.origin = glm::vec4(aperturePoint.x, aperturePoint.y, aperturePoint.z, 0.0f);
	glm::vec3 tmp= glm::normalize((position + ((pointOnPlaneOneUnitAwayFromEye-position) * focalDistance)) - aperturePoint);
	ray.direction = glm::vec4(tmp.x, tmp.y, tmp.z, 40000000000000000000.0f);

}

__host__ __device__ void Camera::UpdateVariables(){
	verticalAxis = normalize(cross(right, forward));
	
	yHelper=verticalAxis * tan((glm::radians(-fieldOfView.y * 0.5f)));
	xHelper= right * tan(glm::radians(fieldOfView.x * 0.5f));
}

__host__ __device__ bool Camera::IsViewable() const{
	return !circularDistribution;
}
__host__ __device__ void Camera::SetCircle(bool cir){
	circularDistribution = cir;
}
__host__ __device__ void Camera::OffsetOrientation(float x, float y){
	right = normalize(glm::rotateY(right, glm::radians(x)));
	forward = normalize(glm::rotateY(forward, glm::radians(x)));

	forward = normalize(glm::rotate(forward, glm::radians(y),right));
}