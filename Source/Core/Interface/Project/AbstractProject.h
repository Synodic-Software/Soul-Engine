#pragma once
#include <filesystem>

class AbstractProject {
	
public:

	AbstractProject();
	virtual ~AbstractProject() noexcept = default;

	const std::filesystem::path& GetDirectory() const;

private:

	std::filesystem::path projectPath_;

};
