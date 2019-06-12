#pragma once

#include <filesystem>


enum class AssetType {
    Image,
    Shader
};

class Asset {
	
public:

	Asset() = default;
	~Asset() = default;

};

class ShaderAsset : public Asset {

public:
	ShaderAsset() = default;
	~ShaderAsset() = default;

};
