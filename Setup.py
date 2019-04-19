import subprocess
import sys
import os

from pathlib import Path


def main():


    print("Updating the Python environment")

    #TODO: Use a venv
    #subprocess.call(["poetry", "install"])
    

    print("Updating the C++ environment")

    buildPath = Path().absolute() / "Build"
    buildPathString = str(buildPath)

    conanFilePath = Path().absolute() / "Tools" / "Conan"
    conanFilePathString = str(conanFilePath)

    #Set the conan remote
    subprocess.call(["conan", "remote", "add", "--force", "bincrafters", "https://api.bintray.com/conan/bincrafters/public-conan"])

    #Create build directory if it does not exist
    if not os.path.exists(buildPath):
        os.makedirs(buildPath)

    #install conan dependencies
    subprocess.call(["conan", "install", conanFilePathString, "-if", buildPathString, "-g", "cmake_multi", "-s", "build_type=Debug", "--build=missing"])
    subprocess.call(["conan", "install", conanFilePathString, "-if", buildPathString, "-g", "cmake_multi", "-s", "build_type=Release", "--build=missing"])

    #set the package to editable, allowing projects to find it globally via Conan and bypass a remote fetch
    subprocess.call(["conan", "editable", "add", conanFilePathString, "SoulEngine/0.0.1@synodic/testing"])

    print("Finished setup!")

main()