#pragma once


class GPUBuffer {

public:
	GPUBuffer();
	~GPUBuffer();

	void* GetData() {
		return data;
	}

protected:
	void* data;

private:
	
};