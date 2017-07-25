#include "Photography/Camera/CUDA/Camera.cuh"
#include <glm/gtx/rotate_vector.hpp>

Camera::Camera() :
	aspectRatio(0),
	position(0.0f, 0.0f, 0.0f),
	forward(0.0f, 0.0f, 1.0f),
	right(1.0f, 0.0f, 0.0f),
	fieldOfView(90.0f, 65.0f),
	aperture(2 * MILLIMETER),
	focalDistance(17 * MILLIMETER)
{
}

Camera::~Camera() {

}


__device__ void Camera::GenerateRay(const uint sampleID, glm::vec3& origin, glm::vec3& direction, curandState& randState) {

	glm::vec2 sample = film.GetSample(sampleID, randState);
	//float angle = TWO_PI * curand_uniform(&randState);
	//float distance = aperture * sqrt(curand_uniform(&randState));


	/*ALTERNATE aperaturPoint
	+ ((cos(angle) * distance) * right) + ((sin(angle) * distance) * verticalAxis)*/

	glm::vec3 aperturePoint = position;

	glm::vec3 pointOnPlaneOneUnitAwayFromEye =
		aperturePoint + forward + (2 * sample.x - 1) * xHelper + (2 * sample.y - 1) * yHelper;

	origin = glm::vec3(aperturePoint.x, aperturePoint.y, aperturePoint.z);
	glm::vec3 tmp = glm::normalize(position + (pointOnPlaneOneUnitAwayFromEye - position) * focalDistance - aperturePoint);
	direction = glm::vec3(tmp.x, tmp.y, tmp.z);

}

void Camera::UpdateVariables() {
	verticalAxis = normalize(cross(right, forward));

	yHelper = verticalAxis * tan((glm::radians(-fieldOfView.y * 0.5f)));
	xHelper = right * tan(glm::radians(fieldOfView.x * 0.5f));
}

void Camera::OffsetOrientation(float x, float y) {
	right = normalize(glm::rotateY(right, glm::radians(x)));
	forward = normalize(glm::rotateY(forward, glm::radians(x)));

	forward = normalize(glm::rotate(forward, glm::radians(y), right));
}