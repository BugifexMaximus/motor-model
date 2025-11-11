cmake_minimum_required(VERSION 3.16)
project(motor_model_cpp_tests LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(FetchContent)

FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

get_filename_component(PROJECT_ROOT "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
set(NATIVE_SRC_DIR "${PROJECT_ROOT}/src/motor_model/_native")

add_library(motor_model_cont_mpc STATIC
  "${NATIVE_SRC_DIR}/continuous_mpc_core.cpp"
  "${NATIVE_SRC_DIR}/continuous_mpc_manager.cpp"
)

target_include_directories(motor_model_cont_mpc
  PUBLIC
    "${NATIVE_SRC_DIR}"
)

target_compile_definitions(motor_model_cont_mpc
  PRIVATE
    GTEST_LANG_CXX17=1
)

enable_testing()

add_executable(test_continuous_mpc
  test_continuous_mpc.cpp
)

target_link_libraries(test_continuous_mpc
  PRIVATE
    motor_model_cont_mpc
    GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(test_continuous_mpc)
