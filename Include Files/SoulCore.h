#pragma once

#include "Engine Core/Object/Object.h"
#include "Engine Core/Object/Character/Character.h"
#include "Engine Core\Scene\Scene.h"
#include <functional>

/* Values that represent render types. */
enum RenderType {SPECTRAL, PATH};
/* Values that represent graphics the pis. */
enum GraphicsAPI{ OPENGL,VULKAN };

/* Soul initialize. */
void SoulInit();
/* Soul run. */
void SoulRun();
/* Soul terminate. */
void SoulTerminate();

/*
 *    Soul signal close.
 *    @param	parameter1	The first parameter.
 */

void SoulSignalClose(int);

/*
 *    Gets delta time.
 *    @return	The delta time.
 */

double GetDeltaTime();

/*
 *    Submit scene.
 *    @param [in,out]	parameter1	If non-null, the first parameter.
 */

void SubmitScene(Scene);

/*
 *    Removes the scene described by parameter1.
 *    @param [in,out]	parameter1	If non-null, the first parameter.
 */

void RemoveScene(Scene);