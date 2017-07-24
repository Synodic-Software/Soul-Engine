#include "CameraManager.h"
#include "Metrics.h"

#include "Algorithms/Morton Code/MortonCode.h"
#include "Algorithms/Data Algorithms/GPU Sort/Sort.h"

#include <memory>

namespace CameraManager {
	namespace detail {
		std::vector<std::unique_ptr<Camera>> cameras;

		std::vector<uint> indicesH;
		uint* indicesD;

		glm::uvec2 maxSize = glm::uvec2(0, 0);
	}

	/*
	 *    Updates the indices described by res.
	 *    @param [in,out]	res	The resource.
	 */

	void UpdateIndices(glm::uvec2& res) {
		uint increaseX = glm::max(0u, res.x - detail::maxSize.x);
		uint increaseY = glm::max(0u, res.y - detail::maxSize.y);

		if (increaseX != 0 && increaseY != 0) {
			detail::maxSize.x += increaseX;
			detail::maxSize.y += increaseY;

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

			//set the static camera variable
			detail::cameras.front()->indicePointer = detail::indicesD;

		}

	}

	void Update() {
		UpdateIndices(detail::maxSize);

		for(auto& camera : detail::cameras) {
			camera.get()->UpdateVariables();
		}
	}

	Camera* AddCamera(glm::uvec2& res) {
		detail::cameras.emplace_back(new Camera());

		Camera* def = detail::cameras.back().get();

		def->film.resolution = res;

		return def;
	}

	void RemoveCamera() {

	}

}