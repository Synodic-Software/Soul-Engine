from conans import ConanFile, CMake, tools
import os

class SoulEngine(ConanFile):

    #Conan vars
    name = "SoulEngine"
    version = "0.0.1"
    license = "GPLv3"
    url = "https://github.com/Synodic-Software/Soul-Engine"
    description = "Soul Engine is a real-time visualization and simulation engine built on the back of CUDA and Vulkan."
    settings = "os", "compiler", "build_type", "arch"
    options = {"shared": [True, False]}
    default_options = {"shared": False}
    generators = "cmake"
    requires = (
        "glfw/3.2.1@bincrafters/stable",
        "boost/1.68.0@conan/stable",
        "glm/0.9.9.1@g-truc/stable",
        "stb/20180214@conan/stable",
    )

    def configureCMake(self):
        
        cmake = CMake(self)
        cmake.definitions["SOME_DEFINITION"] = True
        cmake.configure()

        return cmake

    def build(self):

        cmake = self.configureCMake()
        cmake.build()
        cmake.install()
        
    def package(self):

        cmake = self.configureCMake()
        cmake.install()

    def package_info(self):

        self.cpp_info.libs = ["SoulEngine"]

