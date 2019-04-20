
##############################################
#Tracer

target_sources(${PROJECT_NAME}
    PRIVATE
        Camera/Camera.cpp
        Camera/Camera.h
        Camera/Film/Film.cpp
        Camera/Film/Film.h
        Camera/Film/Filter/Filter.cpp
        Camera/Film/Filter/Filter.h
        Camera/Lens/Lens.h
        Ray/Ray.cpp
        Ray/Ray.h
        Ray/RayEngine.cpp
        Ray/RayEngine.h
        Ray/RayJob.cpp
        Ray/RayJob.h
)

