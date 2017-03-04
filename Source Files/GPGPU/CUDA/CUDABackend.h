#pragma once

#include <vector>


namespace CUDABackend {

	void ExtractDevices(std::vector<int>&);

	int GetCoreCount();

	int GetWarpSize();

	int GetSMCount();

	int GetBlockHeight();

	void Terminate();

	template<typename type>
	void CreateRasterBuffer(type dataType) {
	/*	glGenBuffers(1, &renderBufferA);
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, renderBufferA);
		glBufferData(GL_SHADER_STORAGE_BUFFER,
			originalScreen.x*originalScreen.y * sizeof(dataType),
			NULL, GL_STATIC_DRAW);

		glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

		CudaCheck(cudaGraphicsGLRegisterBuffer(&cudaBuffer
			, renderBufferA
			, cudaGraphicsRegisterFlagsWriteDiscard));


		CudaCheck(cudaGraphicsMapResources(1, &cudaBuffer, 0));
		size_t num_bytes;
		CudaCheck(cudaGraphicsResourceGetMappedPointer((void **)&bufferData, &num_bytes,
			cudaBuffer));*/

	}

}