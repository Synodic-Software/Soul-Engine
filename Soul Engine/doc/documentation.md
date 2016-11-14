
<!--- 
If background color support is added to github, may wwant to use below:
<style>
	background-color: #F0F8FF;
</style>
-->

Soul Engine Documentation
=====
###Please Note this is a work in progress

<br/>
##Table of Contents
1. [Method Calls](#method-calls)

2. [Custom Types](#custom-types)

<br/> <br/>



Method Calls
----

```C++
void SoulInit();
```

**Requires:** None

**Effects:** Initializes SoulEngine

**Returns:** None

<br/>

```C++
void SoulRun();
```

**Requires:** None

**Effects:** Runs SoulEngine

**Returns:** None

<br/>

```C++
GLFWwindow* SoulCreateWindow(int, float, float);
```

**Requires:** None

**Params:**
- int **param1:** param1 desc
- float **param2:** param2 desc
- float **param2:** param2 desc

**Effects:** Crestes a Window

**Returns:** GLFWwindow pointer to newly created window

<br/>

```C++
void SoulSignalClose();
```

**Requires:** A currently open SoulEngine application

**Effects:** Signals a close of open SoulEngine application

**Returns:** None

<br/>

```C++
glm::vec2* GetMouseChange();
```

**Requires:** Mouse change to capture

**Effects:** Signals a close of open SoulEngine application

**Returns:** A pointer to a glm::vec2

<br/>

```C++
bool RequestRenderSwitch(RenderType);
```

**Requires:** A currently open SoulEngine application with previously specified RenderType

**Params:**
- enum RenderType **param1:** How to render the frames.  See [here](#custom-types) for more info on RenderType

**Effects:** Changes RenderType to argument specificed RenderType if possible

**Returns:** True if switch was successful, else false

<br/>

```C++
bool RequestWindowSwitch(WindowType);
```

**Requires:** A currently open SoulEngine application with previously specified WindowType

**Params:**
- enum WindowType **param1:** New type of window.  See [here](#custom-types) for more info on WindowType

**Effects:** Changes WindowType to argument specificed WindowType if possible

**Returns:** True if switch was successful, else false

<br/>

```C++
bool RequestScreenSize(glm::uvec2);
```

**Requires:** A currently open SoulEngine application with previously specified screen size

**Params:**
- glm::uvec2 **param1:** New screen size

**Effects:** Changes screen size to argument specificed screen size if possible

**Returns:** True if change was successful, else false

<br/>

```C++
void SetKey(int, void(*func)(void));
```

**Requires:** None

**Params:**
- int **param1:**Key to bind action to

- void(*func)(void) __param2:__ An action to be bound to this key

**Effects:** Binds the argument provided key and action

**Returns:** None

<br/>

```C++
int GetSetting(std::string);
```

**Requires:** Existing setting to get

**Params:**
- std::string **param1:** Setting to get


**Effects:** Gets specified setting if possible

**Returns:** int desc...

<br/>

```C++
int GetSetting(std::string, int);
```

**Requires:** Existing setting to get

**Params:**
- std::string **param1:** Setting to get

- int **param2:** more info, desc

**Effects:** Gets specified setting if possible

**Returns:** int desc...

<br/>

```C++
void SetSetting(std::string, std::string);
```

**Requires:** None

**Params:**
- std::string **param1:** desc

- std::string **param2:** desc

**Effects:** Sets setting to the specificed value

**Returns:** None

<br/>

```C++
void AddObject(Scene* scene, glm::vec3& globalPos, const char* file, Material* mat);
```

**Requires:** None

**Params:**

- Scene* __scene:__ Scene to add object to

- glm::vec3& __param2:__ Global position coordinates to add obbject at

- const char* __file:__ Filename containing object

- Material* __mat:__ Material of object

**Effects:** Ads object to scene at specified position with specified material

**Returns:** None

<br/>

```C++
void SubmitScene(Scene*);
```

**Requires:** None

**Params:**
- Scene* __param1:__ Scene to set


**Effects:** Submits specified scene

**Returns:** None

<br/>

```C++
void RemoveScene(Scene*);
```

**Requires:** None

**Params:**
- Scene* __param1:__ Scene to remove


**Effects:** Remove specified scene if possible

**Returns:** None

<br/>

```C++
void AddRenderer(Scene*);
```

**Requires:** None

**Params:**
- Scene* __param1:__ Scene to add renderer to


**Effects:** Add renderer to specified scene if possible

**Returns:** None

<br/>

```C++
void RemoveRenderer(Scene*);
```

**Requires:** None

**Params:**
- Scene* __param1:__ Scene to remove renderer from


**Effects:** Remove renderer from specified scene if possible

**Returns:** None

<br/>

```C++
void AddObject(Scene* scene,glm::vec3& globalPos, const char* file, Material* mat);
```

**Requires:** None

**Params:**
- Scene* __scene:__ Scene to add object to
- Scene* __globalPos:__ Global position in scene to add object to
- Scene* __file:__ Name of file containing the object
- Scene* __mat:__ Material of the object


**Effects:** Adds object from file to specified scene at globalPos in material mat if possible

**Returns:** None

<br/>

```C++
void RemoveObject(void*);
```

**Requires:** None

**Params:**
- void* __param1:__ Object to remove


**Effects:** Remove specified object from it's scene if possible

**Returns:** None

<br/> <br/>

Custom Types
----

```C++
typedef enum RenderType {SPECTRAL, PATH};
```

**SPECTRAL**: desc

**PATH**: desc

<br/>

```C++
typedef enum WindowType{ WINDOWED, FULLSCREEN, BORDERLESS };
```

**WINDOWED**: Run application in windowed mode

**FULLSCREEN**: Run application in fullscreen mode

**BORDERLESS**: Run application in borderless mode