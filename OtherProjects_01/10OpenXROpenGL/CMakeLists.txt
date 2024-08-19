include_directories(../openxr)

add_executable(10OpenXROpenGL WIN32 main.cpp)

target_link_libraries( 10OpenXROpenGL OptixUtils glfw sutil_7_sdk
    debug ${CMAKE_SOURCE_DIR}/openxr_loaderd.lib 
    optimized ${CMAKE_SOURCE_DIR}/openxr_loader.lib)

target_link_libraries( 10OpenXROpenGL    
   )
    
