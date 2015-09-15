
#include "Camera.h"
#include <glm/gtc/matrix_transform.hpp>

static const float MaxVerticalAngle = 89.5f; //must be less than 90 to avoid gimbal lock


Camera::Camera() :
	isAttached(false),
	cameraExtract(NULL),
    _position(0,0,0),
    _horizontalAngle(0.0f),
    _verticalAngle(0.0f),
    _fieldOfView(90.0f,65.0f),
    _nearPlane(1*DECIMETER),
    _farPlane(2147483647/2.0f),
    _viewportAspectRatio(16.0f/9.0f)
{	
}

void Camera::AttachCamera(Camera* character){
	isAttached = true;
	cameraExtract = character;
}
void Camera::DetachCamera(){
	isAttached = false;
	cameraExtract = NULL;
}

void Camera::setOrientation(float setVerticle,float setHorizontal){
	DetachCamera();
	 _verticalAngle=setVerticle;
	 _horizontalAngle=setHorizontal;
}
const glm::vec3& Camera::position() const {
	if (isAttached){
		return cameraExtract->position();
	}
    return _position;
}

void Camera::setPosition(const glm::vec3& position) {
	DetachCamera();
    _position = position;
}

void Camera::offsetPosition(const glm::vec3& offset) {
	DetachCamera();
    _position += offset;
}

glm::vec2 Camera::fieldOfView() const {
	if (isAttached){
		return cameraExtract->fieldOfView();
	}
    return _fieldOfView;
}

void Camera::setFieldOfView(glm::vec2 fieldOfView) {
	DetachCamera();
    assert(fieldOfView.x > 0.0f && fieldOfView.x < 180.0f);
	assert(fieldOfView.y > 0.0f && fieldOfView.y < 180.0f);
    _fieldOfView = fieldOfView;
}

float Camera::nearPlane() const {
	if (isAttached){
		return cameraExtract->nearPlane();
	}
    return _nearPlane;
}

float Camera::farPlane() const {
	if (isAttached){
		return cameraExtract->farPlane();
	}
    return _farPlane;
}

void Camera::setNearAndFarPlanes(float nearPlane, float farPlane) {
	DetachCamera();
    assert(nearPlane > 0.0f);
    assert(farPlane > nearPlane);
    _nearPlane = nearPlane;
    _farPlane = farPlane;
}

glm::mat4 Camera::orientation() const {
	if (isAttached){
		return cameraExtract->orientation();
	}
    glm::mat4 orientation;
	orientation = glm::rotate(orientation, glm::radians(_verticalAngle), glm::vec3(1, 0, 0));
    orientation = glm::rotate(orientation, glm::radians(_horizontalAngle), glm::vec3(0,1,0));
    return orientation;
}

void Camera::offsetOrientation( float rightAngle,float upAngle) {
	DetachCamera();
    _horizontalAngle += rightAngle;
    while(_horizontalAngle > 360.0f) _horizontalAngle -= 360.0;
    while(_horizontalAngle < 0.0f) _horizontalAngle += 360.0;

    _verticalAngle += upAngle;

	if (_verticalAngle > MaxVerticalAngle){ 
		_verticalAngle = MaxVerticalAngle; 
	}
	if (_verticalAngle < -MaxVerticalAngle){
		_verticalAngle = -MaxVerticalAngle; 
	}
}

float Camera::viewportAspectRatio() const {
	if (isAttached){
		return cameraExtract->viewportAspectRatio();
	}
    return _viewportAspectRatio;
}

void Camera::setViewportAspectRatio(float viewportAspectRatio) {
	DetachCamera();
    assert(viewportAspectRatio > 0.0);
    _viewportAspectRatio = viewportAspectRatio;
}

glm::vec3 Camera::forward() const {
	if (isAttached){
		return cameraExtract->forward();
	}
    glm::vec4 forward = glm::inverse(orientation()) * glm::vec4(0,0,-1,1);
    return glm::vec3(forward);
}

glm::vec3 Camera::right() const {
	if (isAttached){
		return cameraExtract->right();
	}
    glm::vec4 right = glm::inverse(orientation()) * glm::vec4(1,0,0,1);
    return glm::vec3(right);
}

glm::vec3 Camera::up() const {
	if (isAttached){
		return cameraExtract->up();
	}
    glm::vec4 up = glm::inverse(orientation()) * glm::vec4(0,1,0,1);
    return glm::vec3(up);
}

glm::mat4 Camera::matrix() const {
	if (isAttached){
		return cameraExtract->matrix();
	}
    glm::mat4 camera = glm::perspective(glm::radians(_fieldOfView.y), _viewportAspectRatio, _nearPlane, _farPlane);
    camera *= orientation();
    camera = glm::translate(camera, -_position);
    return camera;
}