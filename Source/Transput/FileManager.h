#pragma once

#include <string>
#include <filesystem>


/*
	Namespace to enclose all file i/o operations.
*/
namespace FileSystem {

	void Read(std::string& filename);
	void Write(std::string& filename);

};