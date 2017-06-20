#pragma once

#include "Engine Core/Object/Object.h"
#include "Engine Core/Object/Character/Character.h"
#include "Engine Core\Scene\Scene.h"
#include <functional>

enum RenderType {SPECTRAL, PATH};
enum GraphicsAPI{ OPENGL,VULKAN };

void SoulInit();
void SoulRun();
void SoulTerminate();

void SoulSignalClose(int);

double GetDeltaTime();

void SubmitScene(Scene*);
void RemoveScene(Scene*);