cmake_minimum_required(VERSION 3.18)
project(mellinger_fuzz_oracle LANGUAGES C CXX)

set(CF_FIRMWARE "" CACHE PATH "Crazyflie firmware checkout")
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(mellinger_fuzz_oracle
    src/wrapper.cpp
    "${CF_FIRMWARE}/src/modules/src/controller/controller_mellinger.c"
    "${CF_FIRMWARE}/src/modules/src/power_distribution_quadrotor.c")
target_include_directories(mellinger_fuzz_oracle PRIVATE
    mocks/common
    mocks/mixer
    "${CF_FIRMWARE}/src/modules/interface"
    "${CF_FIRMWARE}/src/modules/interface/controller"
    "${CF_FIRMWARE}/src/platform/interface"
    "${CF_FIRMWARE}/src/utils/interface"
    "${CF_FIRMWARE}/src/utils/interface/lighthouse"
    "${CF_FIRMWARE}/src/hal/interface")
# Fused multiply-adds change float results compared to the JAX controller
target_compile_options(mellinger_fuzz_oracle PRIVATE -ffp-contract=off)
