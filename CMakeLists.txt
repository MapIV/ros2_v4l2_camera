cmake_minimum_required(VERSION 3.5)
project(v4l2_camera)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()
find_package(CUDA)

# find_package (jpeg REQUIRED)
# TODO: Get rid of this
find_package(LibJpegTurbo REQUIRED)
include_directories(${LIBJPEGTURBO_INCLUDE_DIRS})

# TODO: Make source file paths better
# TODO: Use libjpeg-turbo when CUDA is not present?
# add_library(jpeg_compressor src/jpeg_compressor.cpp
# )
# target_include_directories(jpeg_compressor PUBLIC
#   $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
#   $<INSTALL_INTERFACE:include/v4l2_camera>
# )
# target_link_directories(jpeg_compressor PUBLIC
#   ${SYS_ROOT}/lib/aarch64-linux-gnu
#   ${SYS_ROOT}/usr/lib/aarch64-linux-gnu
#   ${SYS_ROOT}/usr/lib/aarch64-linux-gnu/tegra)
# target_link_libraries(jpeg_compressor
#   ${LIBJPEGTURBO_LIBRARIES}
# )

find_package(OpenCV)

if (EXISTS "/etc/nv_tegra_release")
  set(JETSON TRUE)
  message(STATUS "Jetson platform detected")
else ()
  set(JETSON FALSE)
  message(STATUS "Non-Jetson platform detected")
endif()

if (CUDA_FOUND)
  # Jetson
  if (JETSON)
    find_package(CUDAToolkit)

    include(FetchContent)
    FetchContent_Declare(cuda-api-wrappers
        GIT_REPOSITORY https://github.com/eyalroz/cuda-api-wrappers.git
        GIT_TAG 831666a0bfd1af0f44f4fa234ee2d983d347fcaa # v0.6.1-rc1
    )
    FetchContent_MakeAvailable(cuda-api-wrappers)

    add_library(jpeg_compressor src/acceleration/jpeg_compressor.cpp
      /usr/src/jetson_multimedia_api/samples/common/classes/NvBuffer.cpp
      /usr/src/jetson_multimedia_api/samples/common/classes/NvElement.cpp
      /usr/src/jetson_multimedia_api/samples/common/classes/NvElementProfiler.cpp
      /usr/src/jetson_multimedia_api/samples/common/classes/NvJpegEncoder.cpp
      /usr/src/jetson_multimedia_api/samples/common/classes/NvLogging.cpp
    )
    target_include_directories(jpeg_compressor PUBLIC
      ${CUDA_INCLUDE_DIR}
      ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
      $<INSTALL_INTERFACE:include/v4l2_camera>
      ${SYS_ROOT}/usr/src/jetson_multimedia_api/include
      ${SYS_ROOT}/usr/src/jetson_multimedia_api/include/libjpeg-8b
    )
    target_link_directories(jpeg_compressor PUBLIC
      ${SYS_ROOT}/lib/aarch64-linux-gnu
      ${SYS_ROOT}/usr/lib/aarch64-linux-gnu
      ${SYS_ROOT}/usr/lib/aarch64-linux-gnu/tegra)
    target_link_libraries(jpeg_compressor
      ${CUDA_nppicc_LIBRARY}
      # ${CUDA_nppisu_LIBRARY}
      ${LIBJPEGTURBO_LIBRARIES}
      # /usr/lib/aarch64-linux-gnu/libjpeg.so
      nvjpeg
      cuda-api-wrappers::runtime-and-driver
    )

    target_compile_definitions(jpeg_compressor PUBLIC
      ENABLE_JETSON
    )
    message("Using Jetson Multimedia API")

    cuda_add_library(rectifier
      src/acceleration/rectifier.cpp
      src/acceleration/color_space.cu)
    target_include_directories(rectifier PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
      $<INSTALL_INTERFACE:include/v4l2_camera>
      ${CUDA_INCLUDE_DIR}
      ${OpenCV_INCLUDE_DIRS}
    )
    target_link_directories(rectifier PUBLIC
      ${SYS_ROOT}/lib/aarch64-linux-gnu
      ${SYS_ROOT}/usr/lib/aarch64-linux-gnu
      ${SYS_ROOT}/usr/lib/aarch64-linux-gnu/tegra)
    target_link_libraries(rectifier
      ${CUDA_nppidei_LIBRARY}
      ${CUDA_nppig_LIBRARY}
      ${OpenCV_LIBRARIES}
    )

    # If OpenCV is found
    if (OpenCV_FOUND)
      message(STATUS "OpenCV version: ${OpenCV_VERSION}")
      find_package(cv_bridge REQUIRED)
      target_compile_definitions(rectifier PUBLIC
        ENABLE_OPENCV
      )
    endif()

    if (OpenCV_CUDA_VERSION)
      # Found OpenCV with CUDA support
      message(STATUS "OpenCV CUDA version: ${OpenCV_CUDA_VERSION}")
      target_compile_definitions(rectifier PUBLIC
        ENABLE_OPENCV_CUDA
      )
    endif()
  endif()

  message(STATUS "CUDA Support is enabled in ${PROJECT_NAME}")
  cuda_add_library(v4l2_camera
    src/v4l2_camera.cpp
    src/v4l2_camera_device.cpp
  )
  target_compile_definitions(v4l2_camera PUBLIC
    ENABLE_CUDA
  )
  target_include_directories(v4l2_camera PUBLIC
    ${CUDA_INCLUDE_DIR}
  )
  target_link_libraries(v4l2_camera
    ${CUDA_nppicc_LIBRARY}
    ${CUDA_nppisu_LIBRARY}
    jpeg_compressor
    rectifier
  )
else()
  add_library(v4l2_camera
    src/v4l2_camera.cpp
    src/v4l2_camera_device.cpp)
endif()
rclcpp_components_register_nodes(v4l2_camera "v4l2_camera::V4L2Camera")

target_include_directories(v4l2_camera PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)

ament_target_dependencies(v4l2_camera
  "rclcpp"
  "rclcpp_components"
  "image_transport"
  "camera_info_manager")

ament_target_dependencies(jpeg_compressor
  "rclcpp"
  "rclcpp_components")

ament_target_dependencies(rectifier
  "rclcpp"
  "rclcpp_components"
  "camera_info_manager"
  "image_geometry"
  "cv_bridge")

# TODO: Allow for -Werror
# target_compile_options(v4l2_camera PRIVATE -Werror)
# target_compile_options(v4l2_camera PRIVATE -H)

# Causes the visibility macros to use dllexport rather than dllimport,
# which is appropriate when building the dll but not consuming it.
target_compile_definitions(v4l2_camera PRIVATE "V4L2_CAMERA_BUILDING_LIBRARY")

install(
  DIRECTORY include/
  DESTINATION include
)
install(
  TARGETS v4l2_camera jpeg_compressor rectifier
  EXPORT export_${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

add_executable(v4l2_camera_node src/v4l2_camera_node.cpp)
target_include_directories(v4l2_camera_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_link_libraries(v4l2_camera_node v4l2_camera)
# TODO: Allow for -Werror
# target_compile_options(v4l2_camera_node PRIVATE -Werror)

add_executable(v4l2_camera_compose_test src/v4l2_camera_compose_test.cpp)
target_include_directories(v4l2_camera_compose_test PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_link_libraries(v4l2_camera_compose_test v4l2_camera)
# TODO: Allow for -Werror
# target_compile_options(v4l2_camera_compose_test PRIVATE -Werror)

install(TARGETS v4l2_camera_node v4l2_camera_compose_test
  EXPORT export_${PROJECT_NAME}
  DESTINATION lib/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_export_include_directories(
  include
)
ament_export_targets(
  export_${PROJECT_NAME}
)
ament_export_libraries(
  v4l2_camera
  jpeg_compressor
  rectifier
)

ament_export_dependencies(
  "camera_info_manager"
)

ament_auto_package(INSTALL_TO_SHARE
  launch
)
