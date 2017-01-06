#pragma once

#include "Engine Core/Object/Object.h"
#include "Engine Core/Object/Character/Character.h"
#include "Engine Core\Scene\Scene.h"

enum RenderType {SPECTRAL, PATH};
enum GraphicsAPI{ OPENGL,VULKAN };

void SoulInit();
void SoulRun();
void SoulTerminate();

void SoulSignalClose();

void SetKey(int, void(*func)(void));

void SubmitScene(Scene*);
void RemoveScene(Scene*);

void AddRenderer(Scene*);
void RemoveRenderer(Scene*);

void AddObject(Scene* scene,glm::vec3& globalPos, const char* file, Material* mat);
void RemoveObject(void*);
