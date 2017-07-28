#include "CameraManager.h"
#include "Metrics.h"

#include "Algorithms/Morton Code/MortonCode.h"
#include "Algorithms/Data Algorithms/GPU Sort/Sort.h"

#include <vector>
#include <memory>
#include <iostream>

namespace CameraManager {
	namespace detail {
		std::vector<std::unique_ptr<Camera>> cameras;

		std::vector<uint> indicesH;
		uint* indicesD;

		glm::uvec2 oldMax = glm::uvec2(0, 0);
		glm::uvec2 maxSize = glm::uvec2(0, 0);
	}

	/*
	 *    Updates the indices described by res.
	 *    @param [in,out]	res	The resource.
	 */

	//TODO move the generation of the read pattern to the ray engine.
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

	void Update() {

		//UpdateIndices(detail::maxSize);

		for (auto& camera : detail::cameras) {
			camera->UpdateVariables();
			std::cout << camera->film.resolution.x << std::endl;
		}
	}

	Camera* AddCamera(glm::uvec2& res) {
		detail::cameras.emplace_back(new Camera());

		Camera* def = detail::cameras.back().get();

		//add the resolution
		def->film.resolution = res;
		def->film.resolutionMax = res;

		//update the manger maxSize (for index counting)
		detail::maxSize = glm::max(detail::maxSize, res);

		return def;
	}

	void RemoveCamera() {

	}

}
