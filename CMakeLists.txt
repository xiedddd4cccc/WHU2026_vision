cmake_minimum_required(VERSION 3.16.3)

project(sp_vision)

set(CMAKE_CUDA_ARCHITECTURES 87)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)
message(STATUS "--------------------CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}--------------------")

# 示例
find_package(OpenCV 4 REQUIRED
  COMPONENTS
    core
    imgproc
    highgui
    dnn
    cudawarping
    cudaarithm
    cudaimgproc
)
find_package(fmt REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(spdlog REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(nlohmann_json REQUIRED)
set(CUDAToolkit_ROOT "/usr/local/cuda")
find_package(CUDA REQUIRED)

# ---------------- TensorRT 配置 ----------------
set(TENSORRT_ROOT /usr)
include_directories(${TENSORRT_ROOT}/include)

find_library(NVINFER_LIB nvinfer HINTS ${TENSORRT_ROOT}/lib ${TENSORRT_ROOT}/lib/aarch64-linux-gnu)
find_library(NVINFER_PLUGIN_LIB nvinfer_plugin HINTS ${TENSORRT_ROOT}/lib ${TENSORRT_ROOT}/lib/aarch64-linux-gnu)
find_library(NVONNXPARSER_LIB nvonnxparser HINTS ${TENSORRT_ROOT}/lib ${TENSORRT_ROOT}/lib/aarch64-linux-gnu)
find_library(NVINFER_BUILDER_LIB nvinfer_builder_resource HINTS ${TENSORRT_ROOT}/lib ${TENSORRT_ROOT}/lib/aarch64-linux-gnu)
find_library(NVINFER_RUNTIME_LIB nvinfer_runtime HINTS ${TENSORRT_ROOT}/lib ${TENSORRT_ROOT}/lib/aarch64-linux-gnu)

set(TENSORRT_LIBS
    ${NVINFER_LIB}
    ${NVINFER_PLUGIN_LIB}
    ${NVONNXPARSER_LIB}
    ${NVINFER_BUILDER_LIB}
    ${NVINFER_RUNTIME_LIB}
)

message(STATUS "TensorRT libs: ${TENSORRT_LIBS}")

# ------------------------------------------------

include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR})

add_subdirectory(tools)
add_subdirectory(io)
add_subdirectory(tasks/auto_aim)
add_subdirectory(tasks/auto_buff)
add_subdirectory(tasks/omniperception)

# 全局导出 TensorRT 链接变量
set(TENSORRT_LIBS ${TENSORRT_LIBS} PARENT_SCOPE)


##################calibration################
add_executable(capture_gimbal calibration/capture_gimbal.cpp)
add_executable(capture calibration/capture.cpp)
add_executable(calibrate_camera calibration/calibrate_camera.cpp)
add_executable(calibrate_handeye calibration/calibrate_handeye.cpp)
add_executable(calibrate_robotworld_handeye calibration/calibrate_robotworld_handeye.cpp)
add_executable(split_video calibration/split_video.cpp)

target_link_libraries(capture_gimbal ${OpenCV_LIBS} fmt::fmt tools io)
target_link_libraries(capture ${OpenCV_LIBS} fmt::fmt tools io)
target_link_libraries(calibrate_camera ${OpenCV_LIBS} fmt::fmt yaml-cpp tools)
target_link_libraries(calibrate_handeye ${OpenCV_LIBS} fmt::fmt yaml-cpp tools)
target_link_libraries(calibrate_robotworld_handeye ${OpenCV_LIBS} fmt::fmt yaml-cpp tools)
target_link_libraries(split_video ${OpenCV_LIBS} fmt::fmt tools)

##################tests##################
add_executable(camera_test tests/camera_test.cpp)
add_executable(fire_test tests/fire_test.cpp)
add_executable(detector_video_test tests/detector_video_test.cpp)
add_executable(gimbal_response_test tests/gimbal_response_test.cpp)
add_executable(usbcamera_test tests/usbcamera_test.cpp)
add_executable(handeye_test tests/handeye_test.cpp)
add_executable(dm_test tests/dm_test.cpp)
add_executable(hero tests/hero.cpp)

target_link_libraries(camera_test ${OpenCV_LIBS} fmt::fmt tools io)
target_link_libraries(fire_test ${OpenCV_LIBS} fmt::fmt tools io)
target_link_libraries(detector_video_test ${OpenCV_LIBS} fmt::fmt yaml-cpp tools auto_aim)
target_link_libraries(gimbal_response_test ${OpenCV_LIBS} fmt::fmt yaml-cpp tools io)
target_link_libraries(usbcamera_test ${OpenCV_LIBS} fmt::fmt yaml-cpp tools io)
target_link_libraries(handeye_test ${OpenCV_LIBS} fmt::fmt yaml-cpp tools io auto_aim)
target_link_libraries(dm_test ${OpenCV_LIBS} fmt::fmt yaml-cpp tools io)
target_link_libraries(hero ${OpenCV_LIBS} fmt::fmt yaml-cpp tools auto_aim auto_buff io)


add_executable(gimbal_test tests/gimbal_test.cpp)
target_link_libraries(gimbal_test ${OpenCV_LIBS} fmt::fmt yaml-cpp auto_aim tools io)

# 检查 ROS 环境
find_package(ament_cmake QUIET)
find_package(rclcpp QUIET)
find_package(std_msgs QUIET)
find_package(rosidl_typesupport_cpp QUIET)
find_package(sp_msgs QUIET)
# 只有在找到 ROS 环境时才会添加 sentry 和 publish_test
if(ament_cmake_FOUND AND rclcpp_FOUND AND std_msgs_FOUND AND rosidl_typesupport_cpp_FOUND AND sp_msgs_FOUND)
    # 编译 sentry 和 publish_test
    add_executable(sentry src/sentry.cpp)
    add_executable(sentry_bp src/sentry_bp.cpp)
    add_executable(sentry_debug src/sentry_debug.cpp)
    add_executable(publish_test tests/publish_test.cpp)
    add_executable(subscribe_test tests/subscribe_test.cpp)
    add_executable(topic_loop_test tests/topic_loop_test.cpp)

    target_link_libraries(sentry ${OpenCV_LIBS} fmt::fmt yaml-cpp auto_aim omniperception tools io)
    target_link_libraries(sentry_bp ${OpenCV_LIBS} fmt::fmt yaml-cpp auto_aim omniperception tools io)
    target_link_libraries(sentry_debug ${OpenCV_LIBS} fmt::fmt yaml-cpp auto_aim omniperception tools io)
    target_link_libraries(publish_test ${OpenCV_LIBS} fmt::fmt tools io auto_aim)
    target_link_libraries(subscribe_test ${OpenCV_LIBS} fmt::fmt tools io)
    target_link_libraries(topic_loop_test ${OpenCV_LIBS} fmt::fmt tools io)
else()
    message(STATUS "ROS2 environment not found, skipping ROS2-related code.")
endif()
