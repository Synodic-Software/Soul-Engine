cmake_minimum_required(VERSION 3.11.1 FATAL_ERROR)


##############################################
#Project 

project(SoulEngineTests
	VERSION 0.0.1
	DESCRIPTION "Real-time simulation and rendering engine."
	LANGUAGES CXX
)

#CUDA is optional
enable_language(CUDA)
