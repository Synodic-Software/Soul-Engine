#pragma once

#include <filesystem>


enum class ResourceType {
    Image,
    Shader
};

class Resource {
	
public:

	constexpr Resource(std::string_view);
	~Resource() = default;

    const std::filesystem::path& Path() const;


private:

    std::filesystem::path path_;


};

constexpr Resource::Resource(const std::string_view resourcePath) : 
    path_(resourcePath) 
{

}
