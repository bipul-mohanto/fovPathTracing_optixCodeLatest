OPTIX_add_sample_executable(09HelloRaytracingDX11 target_name
  ${SAMPLES_CUDA_DIR}/helpers.h
  CUDABuffer.h
  LaunchParams.h
  main.cpp
  SampleRenderer.cpp
  SampleRenderer.h
  Dx11Window.h
  stb_image_write.h
  deviceProgram.cu    
  OPTIONS -rdc true
  )

set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /SUBSYSTEM:WINDOWS")

target_link_libraries( ${target_name}
  ${CUDA_LIBRARIES}
  )