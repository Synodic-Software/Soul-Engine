#pragma once

#include "Transput/Resource/ResourceLoader.h"


class SPIRVLoader : public ResourceLoader<SPIRVLoader> {
	
public:

	SPIRVLoader() = default;
	~SPIRVLoader() override = default;

	void Load(const std::string_view&) override;


};

