cmake_minimum_required(VERSION 3.18)
project(CrazyflowMellingerFuzzOracle LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(MELLINGER_EXPECTED_FIRMWARE_COMMIT
    "f45ff8fdce4e73e5ce88ec06eee76309d2fa9a39")
set(MELLINGER_EXPECTED_CONTROLLER_SHA256
    "8942b2cc12b235915dae7f02759396ab0147360e4ca98564f7b2e3b54092d325")
set(MELLINGER_EXPECTED_MIXER_SHA256
    "07017878341e15cc2f70406f3a9cc5ea97cb77acf57aac369205154a3614b1d6")
set(CF_FIRMWARE "" CACHE PATH "Exact transient Crazyflie firmware checkout")

if(CF_FIRMWARE STREQUAL "" OR NOT IS_DIRECTORY "${CF_FIRMWARE}/.git")
    message(FATAL_ERROR
        "CF_FIRMWARE must name the transient verified Git checkout")
endif()

find_package(Git REQUIRED)
execute_process(
    COMMAND "${GIT_EXECUTABLE}" -C "${CF_FIRMWARE}" rev-parse HEAD
    OUTPUT_VARIABLE CF_FIRMWARE_COMMIT
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE CF_GIT_RESULT)
if(NOT CF_GIT_RESULT EQUAL 0 OR
   NOT CF_FIRMWARE_COMMIT STREQUAL MELLINGER_EXPECTED_FIRMWARE_COMMIT)
    message(FATAL_ERROR
        "Firmware commit mismatch: expected "
        "${MELLINGER_EXPECTED_FIRMWARE_COMMIT}, got ${CF_FIRMWARE_COMMIT}")
endif()
execute_process(
    COMMAND "${GIT_EXECUTABLE}" -C "${CF_FIRMWARE}" status --porcelain
    OUTPUT_VARIABLE CF_FIRMWARE_STATUS
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE CF_STATUS_RESULT)
if(NOT CF_STATUS_RESULT EQUAL 0 OR NOT CF_FIRMWARE_STATUS STREQUAL "")
    message(FATAL_ERROR
        "Transient firmware checkout is not clean: ${CF_FIRMWARE_STATUS}")
endif()

set(CF_MELLINGER_SOURCE
    "${CF_FIRMWARE}/src/modules/src/controller/controller_mellinger.c")
set(CF_MIXER_SOURCE
    "${CF_FIRMWARE}/src/modules/src/power_distribution_quadrotor.c")
foreach(REQUIRED_SOURCE "${CF_MELLINGER_SOURCE}" "${CF_MIXER_SOURCE}")
    if(NOT EXISTS "${REQUIRED_SOURCE}")
        message(FATAL_ERROR "Required pinned firmware source missing: ${REQUIRED_SOURCE}")
    endif()
endforeach()
file(SHA256 "${CF_MELLINGER_SOURCE}" CF_MELLINGER_SHA256)
file(SHA256 "${CF_MIXER_SOURCE}" CF_MIXER_SHA256)
if(NOT CF_MELLINGER_SHA256 STREQUAL MELLINGER_EXPECTED_CONTROLLER_SHA256)
    message(FATAL_ERROR
        "controller_mellinger.c SHA-256 mismatch: ${CF_MELLINGER_SHA256}")
endif()
if(NOT CF_MIXER_SHA256 STREQUAL MELLINGER_EXPECTED_MIXER_SHA256)
    message(FATAL_ERROR
        "power_distribution_quadrotor.c SHA-256 mismatch: ${CF_MIXER_SHA256}")
endif()

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
execute_process(
    COMMAND "${Python_EXECUTABLE}" -c
            "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE PYBIND11_DISCOVERY_RESULT)
if(NOT PYBIND11_DISCOVERY_RESULT EQUAL 0 OR pybind11_DIR STREQUAL "")
    message(FATAL_ERROR "pybind11 is not importable with ${Python_EXECUTABLE}")
endif()
find_package(pybind11 CONFIG REQUIRED)

set(CF_COMMON_MOCK_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/mocks/common")
set(CF_MIXER_MOCK_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/mocks/mixer")
set(CF_CONTROLLER_SOURCE_INCLUDE_DIR
    "${CF_FIRMWARE}/src/modules/src/controller")
set(CF_CONTROLLER_INTERFACE_INCLUDE_DIR
    "${CF_FIRMWARE}/src/modules/interface/controller")
set(CF_MODULE_INTERFACE_INCLUDE_DIR
    "${CF_FIRMWARE}/src/modules/interface")
set(CF_PLATFORM_INTERFACE_INCLUDE_DIR
    "${CF_FIRMWARE}/src/platform/interface")
set(CF_UTILS_INTERFACE_INCLUDE_DIR
    "${CF_FIRMWARE}/src/utils/interface")
set(CF_LIGHTHOUSE_INCLUDE_DIR
    "${CF_FIRMWARE}/src/utils/interface/lighthouse")
set(CF_IMU_INCLUDE_DIR "${CF_FIRMWARE}/src/hal/interface")
set(CF_CONTROLLER_INCLUDES
    "${CF_COMMON_MOCK_INCLUDE_DIR}"
    "${CF_MIXER_MOCK_INCLUDE_DIR}"
    "${CF_CONTROLLER_SOURCE_INCLUDE_DIR}"
    "${CF_CONTROLLER_INTERFACE_INCLUDE_DIR}"
    "${CF_MODULE_INTERFACE_INCLUDE_DIR}"
    "${CF_PLATFORM_INTERFACE_INCLUDE_DIR}"
    "${CF_UTILS_INTERFACE_INCLUDE_DIR}"
    "${CF_LIGHTHOUSE_INCLUDE_DIR}"
    "${CF_IMU_INCLUDE_DIR}")
set(CF_MIXER_INCLUDES
    "${CF_COMMON_MOCK_INCLUDE_DIR}"
    "${CF_MIXER_MOCK_INCLUDE_DIR}"
    "${CF_MODULE_INTERFACE_INCLUDE_DIR}"
    "${CF_PLATFORM_INTERFACE_INCLUDE_DIR}"
    "${CF_UTILS_INTERFACE_INCLUDE_DIR}"
    "${CF_LIGHTHOUSE_INCLUDE_DIR}"
    "${CF_IMU_INCLUDE_DIR}")
foreach(INCLUDE_DIR IN LISTS CF_CONTROLLER_INCLUDES CF_MIXER_INCLUDES)
    if(NOT IS_DIRECTORY "${INCLUDE_DIR}")
        message(FATAL_ERROR "Required include directory missing: ${INCLUDE_DIR}")
    endif()
endforeach()

# Only the original controller and raw Legacy-X power distributor are compiled
# as firmware C objects. No battery, cap, motor, stabilizer, or hardware source
# is part of this target.
add_library(crazyflie_mellinger OBJECT "${CF_MELLINGER_SOURCE}")
set_source_files_properties("${CF_MELLINGER_SOURCE}" PROPERTIES LANGUAGE C)
set_target_properties(crazyflie_mellinger PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(crazyflie_mellinger PRIVATE ${CF_CONTROLLER_INCLUDES})

add_library(crazyflie_power_distribution OBJECT "${CF_MIXER_SOURCE}")
set_source_files_properties("${CF_MIXER_SOURCE}" PROPERTIES LANGUAGE C)
set_target_properties(crazyflie_power_distribution PROPERTIES
    POSITION_INDEPENDENT_CODE ON)
target_include_directories(crazyflie_power_distribution PRIVATE
    ${CF_MIXER_INCLUDES})

pybind11_add_module(mellinger_fuzz_oracle MODULE
    src/wrapper.cpp
    $<TARGET_OBJECTS:crazyflie_mellinger>
    $<TARGET_OBJECTS:crazyflie_power_distribution>)
target_include_directories(mellinger_fuzz_oracle PRIVATE
    "${CF_COMMON_MOCK_INCLUDE_DIR}"
    "${CF_MIXER_MOCK_INCLUDE_DIR}")
target_include_directories(mellinger_fuzz_oracle SYSTEM PRIVATE
    "${CF_CONTROLLER_SOURCE_INCLUDE_DIR}"
    "${CF_CONTROLLER_INTERFACE_INCLUDE_DIR}"
    "${CF_MODULE_INTERFACE_INCLUDE_DIR}"
    "${CF_PLATFORM_INTERFACE_INCLUDE_DIR}"
    "${CF_UTILS_INTERFACE_INCLUDE_DIR}"
    "${CF_LIGHTHOUSE_INCLUDE_DIR}"
    "${CF_IMU_INCLUDE_DIR}")

if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(crazyflie_mellinger PRIVATE
        -Wall -Wextra -Wpedantic -fno-fast-math -ffp-contract=off)
    target_compile_options(crazyflie_power_distribution PRIVATE
        -Wall -Wextra -Wpedantic -fno-fast-math -ffp-contract=off)
endif()
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(mellinger_fuzz_oracle PRIVATE
        -Wall -Wextra -Werror -fno-fast-math -ffp-contract=off)
endif()
if(UNIX AND NOT APPLE)
    target_link_libraries(mellinger_fuzz_oracle PRIVATE m)
endif()

target_compile_definitions(mellinger_fuzz_oracle PRIVATE
    MELLINGER_FIRMWARE_COMMIT="${CF_FIRMWARE_COMMIT}"
    MELLINGER_CONTROLLER_SHA256="${CF_MELLINGER_SHA256}"
    MELLINGER_MIXER_SHA256="${CF_MIXER_SHA256}"
    MELLINGER_C_COMPILER_ID="${CMAKE_C_COMPILER_ID}"
    MELLINGER_C_COMPILER_VERSION="${CMAKE_C_COMPILER_VERSION}")

message(STATUS "Transient firmware commit: ${CF_FIRMWARE_COMMIT}")
message(STATUS "Controller SHA-256: ${CF_MELLINGER_SHA256}")
message(STATUS "Mixer SHA-256: ${CF_MIXER_SHA256}")
