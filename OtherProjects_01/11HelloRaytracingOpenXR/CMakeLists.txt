include_directories(../openxr)

OPTIX_add_sample_executable(11HelloRaytracingOpenXR target_name
  ${SAMPLES_CUDA_DIR}/helpers.h
  CUDABuffer.h
  LaunchParams.h
  main.cpp
  OptiXRenderer.h
  SampleRenderer.cpp
  SampleRenderer.h
  stb_image_write.h
  deviceProgram.cu    
  OPTIONS -rdc true
  )

target_link_libraries( 11HelloRaytracingOpenXR ${CUDA_LIBRARIES}
    debug ${CMAKE_SOURCE_DIR}/openxr_loaderd.lib 
    optimized ${CMAKE_SOURCE_DIR}/openxr_loader.lib)
    
