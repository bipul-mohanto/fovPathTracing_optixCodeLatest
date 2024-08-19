add_executable(08DirectX11Sample WIN32 main.cpp data.h)

add_custom_command(TARGET 08DirectX11Sample POST_BUILD        # Adds a post-build event to MyTest
    COMMAND ${CMAKE_COMMAND} -E copy                          # which executes "cmake - E copy_if_different..."
        ${PROJECT_SOURCE_DIR}/08DirectX11Sample/shaders.hlsl                   # <--this is in-file
        $<TARGET_FILE_DIR:08DirectX11Sample>)                 # <--this is out-file path
