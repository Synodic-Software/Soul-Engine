#include "CameraManager.h"
#include "Metrics.h"

#include "Algorithms/Morton Code/MortonCode.h"
#include "Algorithms/Data Algorithms/GPU Sort/Sort.h"

#include <vector>
#include <iostream>

#include "GPGPU/GPUManager.h"

namespace CameraManager {

	namespace detail {

		GPUBuffer<Camera>* cameras;
		std::list<uint> cameraIDs;

		uint globalID = 0;

		std::vector<uint> indicesH;
		uint* indicesD;

		glm::uvec2 oldMax = glm::uvec2(0, 0);
		glm::uvec2 maxSize = glm::uvec2(0, 0);
	}

	/*
	 *    Updates the indices described by res.
	 *    @param [in,out]	res	The resource.
	 */

	 //TODO move the generation of the read pattern to the GPUBuffer.
	void UpdateIndices(glm::uvec2& res) {

		if (detail::oldMax != detail::maxSize) {

			uint size = detail::maxSize.x*detail::maxSize.y;

			detail::indicesH.resize(size);
			std::vector<uint64> codes(size);

			//TODO parallelize
			//generate the indices and codes
			for (uint j = 0; j < detail::maxSize.y; ++j) {

				for (uint i = 0; i < detail::maxSize.x; ++i) {

					uint64 code = MortonCode::Calculate64_2D(glm::uvec2(i, j));

					uint index = i*detail::maxSize.y + j;
					codes[index] = code;
					detail::indicesH[index] = index;
				}
			}

			//sort
			detail::indicesD = Sort::Calculate(size, codes.data(), detail::indicesH.data());

			//TODO set the read pattern for the ray buffer

			detail::oldMax = detail::maxSize;
		}

	}

	void Initialize() {
		detail::cameras = new GPUBuffer<Camera>(GPUManager::GetBestGPU());
	}

	void Terminate() {
		delete detail::cameras;
	}

	void Update() {

		//UpdateIndices(detail::maxSize);
		for (int i = 0; i < detail::cameras->size(); ++i) {
			Camera& camera = (*detail::cameras)[i];
			camera.UpdateVariables();
		}
	}

	//TODO return a pointer or ID, not
	uint AddCamera(glm::uvec2& res) {

		uint id = detail::globalID++;

		detail::cameras->push_back(Camera());
		detail::cameraIDs.push_back(id);

		Camera* def = detail::cameras->back();

		//add the resolution
		def->film.resolution = res;
		def->film.resolutionMax = res;

		//update the manger maxSize (for index counting)
		detail::maxSize = glm::max(detail::maxSize, res);

		return id;
	}

	GPUBuffer<Camera>* GetCameraBuffer() {
		return detail::cameras;
	}

	Camera* GetCamera(uint id) {

		int index = std::distance(
			detail::cameraIDs.begin(), 
			std::find(
				detail::cameraIDs.begin(), 
				detail::cameraIDs.end(),
				id)
		);

		return (*detail::cameras).begin()+index;
	}


	//TODO add functionality (AddCamera return needs revisit)
	void RemoveCamera() {

	}

}
