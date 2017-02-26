#pragma once

enum Backend {CUDA, OpenCL};

namespace GPUManager {

	void ExtractDevices();

}