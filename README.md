# What is Soul Engine?
Soul Engine is a physically based renderer and engine for real-time applications on any 
system with CUDA (primary) or OpenCL (secondary) support. 

![Winged Victory Model](WingedVictoryBroken.png)
(This is image is produced after 10 seconds of accumulation on a using a 750TI)

#Features
These are planned features (partial/incomplete implementations) for aid in the vision of the final application.

  -Spectral bidirectional path tracer
  
  -GPU physics pipeline which includes an FEM solver for all objects
  
  -A ray engine which handles ray requests and processes them in bulk.
  
  -Sound path tracing with a multi-listener setup.
  
  -Shader system allowing for immediate artist controlled changes.
  
  -Fiber implementation for task based CPU processing.
  
  -Calculation determinism allowing for lockstep networking.
  
# Current Status
Completed features:

  -The bulk of the main ray engine is complete allowing for any code to request a parrallel ray job near rendertime. Jobs are coalesced, at the cost of minor overhead, and sent into the scene to collect information. 
  
  -Basic (and slow) path tracing is available using only diffuse materials.
  
  -LBVH implementation is complete with many slow and missing features

Ways to interact with Soul Engine beyond this repository are currently being investigated.
For your propiertery purposes, an alternate license will be also made available once the project is in a decent enough shape.


