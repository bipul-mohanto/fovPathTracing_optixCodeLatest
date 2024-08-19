#include "SampleRenderer.h"

#include <sutil/sutil.h>
#include <sutil/CUDAOutputBuffer.h>

#include <sutil/GLDisplay.h>
#include <GLFW/glfw3.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

bool resize_dirty = false;
bool minimized = false;
int2 fbSize;

static void keyCallback(GLFWwindow* window, int32_t key, int32_t /*scancode*/, int32_t action, int32_t /*mods*/)
{
    if (action == GLFW_PRESS)
    {
        if (key == GLFW_KEY_Q || key == GLFW_KEY_ESCAPE)
        {
            glfwSetWindowShouldClose(window, true);
        }
    }
    else if (key == GLFW_KEY_G)
    {
        // toggle UI draw
    }
}

static void windowIconifyCallback(GLFWwindow* window, int32_t iconified)
{
    minimized = (iconified > 0);
}

static void windowSizeCallback(GLFWwindow* window, int32_t res_x, int32_t res_y)
{
    // Keep rendering at the current resolution when the window is minimized.
    if (minimized)
        return;

    // Output dimensions must be at least 1 in both x and y.
    sutil::ensureMinimumSize(res_x, res_y);

    fbSize = make_int2(res_x, res_y);
    resize_dirty = true;
}

void displaySubframe(sutil::CUDAOutputBuffer<uint32_t>& output_buffer, sutil::GLDisplay& gl_display, GLFWwindow* window)
{
    // Display
    int framebuf_res_x = 0;  // The display's resolution (could be HDPI res)
    int framebuf_res_y = 0;  //
    glfwGetFramebufferSize(window, &framebuf_res_x, &framebuf_res_y);
    gl_display.display(
        output_buffer.width(),
        output_buffer.height(),
        framebuf_res_x,
        framebuf_res_y,
        output_buffer.getPBO()
    );
}

extern "C" int main(int ac, char** av)
{
    try {
        SampleRenderer sample;

        fbSize = make_int2(1200, 1024);
        sample.resize(fbSize);
        sample.render();

        std::vector<uint32_t> pixels(fbSize.x * fbSize.y);        

        GLFWwindow* window = sutil::initUI("optixPathTracer", fbSize.x, fbSize.y);
 
        glfwSetWindowSizeCallback(window, windowSizeCallback); 
        glfwSetKeyCallback(window, keyCallback);

        //
        // Render loop
        //
        {
            sutil::CUDAOutputBufferType output_buffer_type = sutil::CUDAOutputBufferType::GL_INTEROP;

            sutil::CUDAOutputBuffer<uint32_t> output_buffer(
                output_buffer_type,
                fbSize.x,
                fbSize.y
            );
            
            output_buffer.setStream(sample.stream);
            sutil::GLDisplay gl_display;

            std::chrono::duration<double> state_update_time(0.0);
            std::chrono::duration<double> render_time(0.0);
            std::chrono::duration<double> display_time(0.0);

            do
            {
                auto t0 = std::chrono::steady_clock::now();
                glfwPollEvents();

                if (resize_dirty){
                    sample.resize(fbSize);
                    output_buffer.resize(fbSize.x, fbSize.y);
                    resize_dirty = false;
                }
                auto t1 = std::chrono::steady_clock::now();
                state_update_time += t1 - t0;
                t0 = t1;

                uint32_t* result_buffer_data = output_buffer.map();
                sample.launchParams.colorBuffer = result_buffer_data;
                sample.render();
                output_buffer.unmap();
                t1 = std::chrono::steady_clock::now();
                render_time += t1 - t0;
                t0 = t1;

                displaySubframe(output_buffer, gl_display, window);
                t1 = std::chrono::steady_clock::now();
                display_time += t1 - t0;

                sutil::displayStats(state_update_time, render_time, display_time);

                glfwSwapBuffers(window);

            } while (!glfwWindowShouldClose(window));
            CUDA_SYNC_CHECK();
        }

        sutil::cleanupUI(window);

        /*sample.downloadPixels(pixels.data());

        sutil::ImageBuffer buffer;
        buffer.data = pixels.data();
        buffer.width = fbSize.x;
        buffer.height = fbSize.y;
        buffer.pixel_format = sutil::BufferImageFormat::UNSIGNED_BYTE4;
        sutil::displayBufferWindow(*av, buffer);

        const std::string fileName = "osc_example2.png";
        stbi_write_png(fileName.c_str(), fbSize.x, fbSize.y, 4,
            pixels.data(), fbSize.x * sizeof(uint32_t));
        std::cout 
            << std::endl
            << "Image rendered, and saved to " << fileName << " ... done." << std::endl            
            << std::endl;*/
    }
    catch (std::runtime_error& e) {
        std::cout  << "FATAL ERROR: " << e.what() << std::endl;
        exit(1);
    }
    return 0;
}