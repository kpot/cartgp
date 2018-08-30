cmake_minimum_required(VERSION 3.3)
project(cartgp)

include(CTest)

add_subdirectory(cpp/libs/pybind11)

add_library(cartgp INTERFACE)
target_compile_features(cartgp INTERFACE cxx_std_11)
target_include_directories(cartgp INTERFACE cpp/include)

pybind11_add_module(pycartgp cpp/src/cartgp/python_binding.cpp)
target_compile_features(pycartgp PRIVATE cxx_std_11)
target_link_libraries(pycartgp PRIVATE cartgp)

if (BUILD_TESTING)
    set(TESTS
            test_cartgp)
    foreach(TARGET ${TESTS})
        add_executable(${TARGET} cpp/src/tests/${TARGET}.cpp)
        target_compile_features(${TARGET} PRIVATE cxx_std_14)
        target_include_directories(${TARGET} PRIVATE cpp/libs/Catch2)
        target_link_libraries(${TARGET} PRIVATE cartgp)
        add_test(NAME ${TARGET} COMMAND ${TARGET})
    endforeach()
endif()
