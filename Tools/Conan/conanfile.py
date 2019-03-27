from conans import ConanFile, CMake, tools
from pathlib import Path

import os

class SoulEngine(ConanFile):

    name = "SoulEngine"
    version = "0.0.1"
    author = "Synodic Software"
    license = "GPLv3"
    url = "https://github.com/Synodic-Software/Soul-Engine"
    description = "Soul Engine is a real-time visualization and simulation engine built on the back of CUDA, OpenCL, and Vulkan."

    settings = "os", "compiler", "build_type", "arch"
    options = {"shared": [True, False]}
    default_options = {"shared": False}

    generators = "cmake"

    scm = {
        "type" : "git",
        "url" : "auto",
        "subfolder": "../..",
        "revision" : "auto"
    }

    no_copy_source = True

    requires = (
        "glfw/3.2.1@bincrafters/stable",    
        "boost/1.69.0@conan/stable",
        "glm/0.9.9.1@g-truc/stable",
        "stb/20180214@conan/stable",     
    )

    def configureCMake(self):
        
        cmake = CMake(self)
        cmake.configure(source_folder="../..")

        return cmake


    def build(self):

        cmake = self.configureCMake()
        cmake.build()


    def package(self):

        cmake = self.configureCMake()

        #TODO: no_copy_source causes cmake to install twice see https://docs.conan.io/en/latest/howtos/cmake_install.html
        cmake.install()


    def package_info(self):

        self.cpp_info.libs = ["SoulEngine"]

        projectRoot = Path('.') / ".." / ".."
        self.cpp_info.includedirs = [str(projectRoot / "Includes")]

        #CMAKE environment variables
        self.user_info.ENGINE_PATH = str(Path().absolute().parents[1])