# Table of Contents
1. [Description](#description)

2. [Sample Usage](#sample-usage)

3. [Public Methods](#public-methods)

     1. [Fiber Priorities](#fiberpriorities)

4. [Enumeration Types](#enumeration-types)

# Description
A lightweight fiber implementation used to efficiently execute many tasks in parallel.  This provides a better alternative to threads.

# Sample Usage

```C++
#include "Multithreading/Scheduler.h"
//More Dependencies depending on use case.  See below link.

//setup the multithreader
Scheduler::Init();

//open the config file for the duration of the runtime
Scheduler::AddTask(LAUNCH_IMMEDIATE, FIBER_HIGH, false, []() {
	Settings::Read("config.ini",TEXT);
});

//extract all available GPU devices
Scheduler::AddTask(LAUNCH_IMMEDIATE, FIBER_HIGH, false, []() {
	GPUManager::ExtractDevices();
});

//set the error callback
Scheduler::AddTask(LAUNCH_IMMEDIATE, FIBER_HIGH, true, []() {
	glfwSetErrorCallback([](int error, const char* description) {
		S_LOG_FATAL("GLFW Error occured, Error ID:", error, " Description:", description);
	});
});

//Init glfw context for Window handling
int didInit;
Scheduler::AddTask(LAUNCH_IMMEDIATE, FIBER_HIGH, true, [&didInit]() {
	didInit = glfwInit();
});

Scheduler::Block();

//init main Window
Scheduler::AddTask(LAUNCH_IMMEDIATE, FIBER_HIGH, false, []() {
	WindowManager::Init(&running);
});

if (!didInit) {
	S_LOG_FATAL("GLFW did not initialize");
}

Settings::Get("Engine.Engine_Refresh_Rate",60.0,&engineRefreshRate);

Scheduler::Block();
```

See [here](https://github.com/Behemyth/Soul-Engine/blob/master/Source%20Files/Engine%20Core/SoulCore.cpp) for more example usage.

# Public Methods 

```C++
void InitPointers();
```
**Effects**: Initializes all fiber_specific_ptrs if they haven't been initialized.

<br>

```C++
void Init();
```
**Effects**: Initialize the multithreaded scheduler.

<br>

```C++
void Terminate();
```
**Effects**: Terminate the multithreaded scheduler.

<br>

```C++
template<typename Fn,
		typename ... Args>
		void AddTask(FiberPolicy policy, FiberPriority priority, bool runsOnMain, Fn && fn, Args && ... args);
```
**Params:**
- FiberPolicy **policy**: The fiber policy after running the segment.
- FiberPriority **priority**: Fiber execution priority.
- bool **runsOnMain**: Requirement that this function runs on the main thread.
- Fn **fn**: Lambda function to be executed.
- Args **...args**: Arguments to fn.

**Effects:** Adds a task to the fiber scheduler to be executed concurrently.

<br>

```C++
void Block();
```
**Effects**: Blocks the fiber until all tasks with the LAUNCH_IMMEDIATE policy have been executed.

<br>

```C++
void Defer();
```
**Effects**: Yields the current fiber to the scheduler.

<br>

```C++
bool Running() const;
```
**Returns**: Returns the running state of the scheduler. Useful for functions that want to run the lifespan of the engine.

<br>

## FiberPriorities
**Description:** Property class for the custom scheduler.

**Notes:** Has public inheritance from boost::fibers::fiber_properties.

<br>

```C++
FiberProperties(boost::fibers::context * context)
```
**Params:**
 - boost::fibers::context* **context**: context to initialize superclass

**Effects:** Creates a new FiberPriorities object.

**Returns:** The new FiberPriorities object.

<br>

```C++
int GetPriority() const;
```

**Returns:** The priority of the fiber.

<br>

```C++
bool RunOnMain() const;
```

**Returns:** Whether or not the fiber should run on main.

<br>

```C++
void SetPriority(int p, bool m);
```
**Params:**
- int **p**: The new priority of the fiber.
- bool **m**: Whether the fiber should now run on main

**Effects:** Shares changes is one or both parameters do not match current values (via notify() ).

<br>

# Enumeration Types

```C++
enum FiberPolicy { LAUNCH_IMMEDIATE, LAUNCH_CONTINUE };
```
- **LAUNCH_IMMEDIATE**: Run the fiber immediatly with no context switch.
Use Case: You will execute 100 tasks and will wait till they complete.
- **LAUNCH_CONTINUE**: Keep the current context and add the fiber to the queue.
Use Case: You want to process other things while this function gets executed elsewhere.

<br>

```C++
enum FiberPriority { FIBER_HIGH, FIBER_LOW };
```
- **FIBER_HIGH**: A high priority task.
- **FIBER_LOW**: A comparatively low priority task.

