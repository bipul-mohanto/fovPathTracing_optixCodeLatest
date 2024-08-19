OPTIX_add_sample_executable(02HelloRaytracing target_name
  ${SAMPLES_CUDA_DIR}/helpers.h
  CUDABuffer.h
  LaunchParams.h
  main.cpp
  SampleRenderer.cpp
  SampleRenderer.h
  stb_image_write.h
  deviceProgram.cu    
  OPTIONS -rdc true
  )

target_link_libraries( ${target_name}
  ${CUDA_LIBRARIES}
  )