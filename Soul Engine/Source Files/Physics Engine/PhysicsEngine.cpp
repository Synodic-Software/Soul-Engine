#include "PhysicsEngine.h"
#include "CUDA/PhysicsEngine.cuh"


void PhysicsEngine::Process(const Scene* scene){

	ProcessScene(scene);
}
