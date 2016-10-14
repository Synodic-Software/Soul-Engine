#pragma once

#include "Engine Core/Object/Object.h"
#include "Engine Core/Object/Character/Character.h"
#include "Engine Core\Scene\Scene.h"


typedef enum RenderType {SPECTRAL, PATH};
typedef enum WindowType{ WINDOWED, FULLSCREEN, BORDERLESS };

void SoulInit();
void SoulRun();
//WindowType, RenderType
GLFWwindow* SoulCreateWindow(int, float,float);
void SoulSignalClose();
glm::vec2* GetMouseChange();
//void Run();

bool RequestRenderSwitch(RenderType);
bool RequestWindowSwitch(WindowType);
bool RequestScreenSize(glm::uvec2);

void SetKey(int, void(*func)(void));

int GetSetting(std::string);
int GetSetting(std::string, int);
void SetSetting(std::string, std::string);

void AddObject(Scene* scene,glm::vec3& globalPos, const char* file, Material* mat);
void SubmitScene(Scene*);
void RemoveScene(Scene*);
void AddRenderer(Scene*);
void RemoveRenderer(Scene*);

bool IsRunning();
void RemoveObject(void*);
