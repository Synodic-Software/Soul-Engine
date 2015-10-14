#include "Engine Core/Camera/CUDA/Camera.cuh"

CUDA_FUNCTION Camera::Camera() :
    position(0.0f,0.0f,0.0f),
	forward(0.0f,0.0f,1.0f),
	right(1.0f, 0.0f, 0.0f),
	fieldOfView(90.0f,65.0f),
	aperture(2*MILLIMETER),
	focalDistance(17*MILLIMETER),
	circularDistribution(false)
{	
}

CUDA_FUNCTION  Camera::~Camera(){

}

CUDA_FUNCTION void Camera::SetAspect(float newA){
	aspectRatio = newA;
}
CUDA_FUNCTION float Camera::GetAspect(){
	return aspectRatio;
}

CUDA_FUNCTION glm::vec3 Camera::Position() const {
    return position;
}

CUDA_FUNCTION void Camera::SetPosition(const glm::vec3& positionN) {
    position = positionN;
}

CUDA_FUNCTION void Camera::OffsetPosition(const glm::vec3& offset) {
    position += offset;
}

CUDA_FUNCTION glm::vec2 Camera::FieldOfView() const{
    return fieldOfView;
}
CUDA_FUNCTION void Camera::SetFieldOfView(glm::vec2 fieldOfView) {
    fieldOfView = fieldOfView;
}



CUDA_FUNCTION glm::vec3 Camera::Forward() const {
    return forward;
}
CUDA_FUNCTION void Camera::SetForward(glm::vec3& forN){
	forward = forN;
}

CUDA_FUNCTION glm::vec3 Camera::Right() const {
    return right;
}
CUDA_FUNCTION void Camera::SetRight(glm::vec3& rightn) {
	right = rightn;
}

CUDA_FUNCTION Ray Camera::SetupRay(uint& index, uint& n, thrust::default_random_engine& rng, thrust::uniform_real_distribution<float>& uniformDistribution){

	int y = int(index / resolution.y);
	int x = index - (y*resolution.y);

	// generate random jitter offsets for supersampled antialiasing
	float jitterValueX = uniformDistribution(rng) - 0.5f;
	float jitterValueY = uniformDistribution(rng) - 0.5f;

	// compute important values
	forward = normalize(forward); // view is already supposed to be normalized, but normalize it explicitly just in case.
	glm::vec3 horizontalAxis = right;
	horizontalAxis = normalize(horizontalAxis); // Important!
	glm::vec3 verticalAxis = glm::cross(horizontalAxis, forward);
	verticalAxis = normalize(verticalAxis); // verticalAxis is normalized by default, but normalize it explicitly just for good measure.

	// compute point on image plane
	glm::vec3 middle = position + forward;
	glm::vec3 horizontal = horizontalAxis * glm::tan(fieldOfView.x * 0.5f * (PI / 180)); // Now treating FOV as the full FOV, not half, so I multiplied it by 0.5. I also normzlized A and B, so there's no need to divide by the length of A or B anymore. Also normalized view and removed lengthOfView. Also removed the cast to float.
	glm::vec3 vertical = verticalAxis * glm::tan(-fieldOfView.y * 0.5f * (PI / 180)); // Now treating FOV as the full FOV, not half, so I multiplied it by 0.5. I also normzlized A and B, so there's no need to divide by the length of A or B anymore. Also normalized view and removed lengthOfView. Also removed the cast to float.

	float sx = (jitterValueX + x) / (resolution.x - 1);
	float sy = (jitterValueY + y) / (resolution.y - 1);

	glm::vec3 pointOnPlaneOneUnitAwayFromEye = middle + (((2 * sx) - 1) * horizontal) + (((2 * sy) - 1) * vertical);

	// move and resize image plane based on focalDistance
	// could also incorporate this into the original computations of the point
	glm::vec3 pointOnImagePlane = position + ((pointOnPlaneOneUnitAwayFromEye - position) * focalDistance); // Important for depth of field!

	// now compute the point on the aperture (or lens)
	glm::vec3 aperturePoint;
		// generate random numbers for sampling a point on the aperture
		float random1 = uniformDistribution(rng);
		float random2 = uniformDistribution(rng);

		// sample a point on the circular aperture
		float angle = TWO_PI * random1;
		float distance = aperture * sqrt(random2);

		float apertureX = cos(angle) * distance;
		float apertureY = sin(angle) * distance;

		aperturePoint = position + (apertureX * horizontalAxis) + (apertureY * verticalAxis);

	//aperturePoint = renderCamera.position;
	glm::vec3 apertureToImagePlane = pointOnImagePlane - aperturePoint;

	Ray ray = Ray(aperturePoint, normalize(apertureToImagePlane));

	return ray;
}

CUDA_FUNCTION bool Camera::IsViewable() const{
	return !circularDistribution;
}
CUDA_FUNCTION void Camera::SetCircle(bool cir){
	circularDistribution = cir;
}
CUDA_FUNCTION void Camera::SetResolution(glm::uvec2 res){
	resolution = res;
}
CUDA_FUNCTION glm::uvec2 Camera::GetResolution(){
	return resolution;
}
CUDA_FUNCTION void Camera::OffsetOrientation(float x, float y){
	right = glm::rotateX(right, glm::radians(x));
	forward = glm::rotateX(forward, glm::radians(x));

	right = glm::rotateY(right, glm::radians(y));
	forward = glm::rotateY(forward, glm::radians(y));
}