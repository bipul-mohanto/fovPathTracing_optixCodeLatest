OPTIX_add_sample_executable( 01HelloRaytracing target_name
  ${SAMPLES_CUDA_DIR}/helpers.h
  draw_solid_color.cu
  helloRaytracing.cpp
  helloRaytracing.h
  
  OPTIONS -rdc true
  )

target_link_libraries( ${target_name}
  ${CUDA_LIBRARIES}
  )