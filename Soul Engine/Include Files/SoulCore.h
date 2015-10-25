#pragma once

#include "Engine Core/Object/Object.cuh"
#include "Engine Core/Object/Character/Character.h"

typedef enum RenderType {SPECTRAL, PATH};
typedef enum WindowType{ WINDOWED, FULLSCREEN, BORDERLESS };

void SoulInit();

//WindowType, RenderType
void SoulCreateWindow(WindowType, RenderType);
void SoulTerminate();
glm::vec2* GetMouseChange();
//void Run();

bool RequestRenderSwitch(RenderType);
bool RequestWindowSwitch(WindowType);
bool RequestScreenSize(glm::uvec2);

std::string GetSetting(std::string);
void SetSetting(std::string, std::string);
void AddObject(Object*);
void RemoveObject(Object*);
