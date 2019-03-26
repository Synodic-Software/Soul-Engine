#pragma once

#include "Transput/Resource/AbstractResourceLoader.h"

class SPIRVLoader: public AbstractResourceLoader<SPIRVLoader> {
	
public:

	SPIRVLoader() = default;
	~SPIRVLoader() override = default;

	void Load(const std::string_view&) override;

};

/*
//set the static for this class
template <>
std::string_view AbstractResourceLoader<SPIRVLoader>::extensions_ [] = { 
	"spv" 
};*/
