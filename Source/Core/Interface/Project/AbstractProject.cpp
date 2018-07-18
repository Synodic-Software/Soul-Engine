#include "AbstractProject.h"

AbstractProject::AbstractProject() {
	
}

const std::filesystem::path& AbstractProject::GetDirectory() const {
	return projectPath_;
}