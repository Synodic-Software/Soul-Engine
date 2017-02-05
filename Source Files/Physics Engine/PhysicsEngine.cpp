#include "PhysicsEngine.h"
#include "CUDA/PhysicsEngine.cuh"

//this just processes a scene with the CUDA physics engine
void PhysicsEngine::Process(const Scene* scene){

	ProcessScene(scene);
}
