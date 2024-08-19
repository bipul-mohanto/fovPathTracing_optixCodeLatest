include_directories(../openxr)

add_executable(07OpenXRSample WIN32 main.cpp)

target_link_libraries( 07OpenXRSample 
    debug ${CMAKE_SOURCE_DIR}/openxr_loaderd.lib 
    optimized ${CMAKE_SOURCE_DIR}/openxr_loader.lib)
