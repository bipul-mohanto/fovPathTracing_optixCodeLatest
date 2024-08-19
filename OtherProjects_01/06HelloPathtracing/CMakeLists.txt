OPTIX_add_sample_executable(06HelloPathtracing target_name
  ${SAMPLES_CUDA_DIR}/helpers.h

  LaunchParams.h
  main.cpp

  SimplePathtracer.cpp
  SimplePathtracer.h   

  Model.cpp
  Model.h

  OptixDenoiser.h
  OptixDenoiser.cpp

  Disney.cuh
  PathTracer.cu
  BSSRDF.cuh
  BSDF.cuh

  OPTIONS -rdc true
  )

target_link_libraries( ${target_name}
  ${CUDA_LIBRARIES}
  toneMap
  OptixUtils
  )