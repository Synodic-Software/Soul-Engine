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

<br/> <br/>

Custom Types
----

```C++
typedef enum RenderType {SPECTRAL, PATH};
```

**SPECTRAL**: desc

**PATH**: desc

<br/>