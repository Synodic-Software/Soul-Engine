#pragma once

#include "Engine Core/Object/Object.h"
#include "Engine Core/Object/Character/Character.h"

typedef enum RenderType {SPECTRAL, PATH};
typedef enum WindowType{ WINDOWED, FULLSCREEN, BORDERLESS };

void SoulInit();
void SoulRun();
//WindowType, RenderType
void SoulCreateWindow(WindowType, RenderType);
void SoulTerminate();
glm::vec2* GetMouseChange();
//void Run();

bool RequestRenderSwitch(RenderType);
bool RequestWindowSwitch(WindowType);
bool RequestScreenSize(glm::uvec2);

void SetKey(int, void(*func)(void));

std::string GetSetting(std::string);
void SetSetting(std::string, std::string);
void AddObject(glm::vec3& globalPos, const char* file, Material* mat);
bool IsRunning();
void RemoveObject(void*);
