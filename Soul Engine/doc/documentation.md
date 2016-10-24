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