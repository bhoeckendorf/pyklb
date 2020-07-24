cmake_minimum_required(VERSION 3.11.0)
project(pyklb)

find_package(PythonExtensions REQUIRED)
find_package(Cython REQUIRED)
find_package(NumPy REQUIRED)


# Update libklb git submodule,
# adapted from https://cliutils.gitlab.io/modern-cmake/chapters/projects/submodule.html
find_package(Git REQUIRED)
message(STATUS "Submodule update")
execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                RESULT_VARIABLE GIT_SUBMOD_RESULT)
if(NOT GIT_SUBMOD_RESULT EQUAL "0")
    message(FATAL_ERROR "git submodule update --init failed with ${GIT_SUBMOD_RESULT}")
endif()

if(NOT EXISTS "${PROJECT_SOURCE_DIR}/libklb/CMakeLists.txt")
    message(FATAL_ERROR "The submodules were not downloaded! GIT_SUBMODULE was turned off or failed. Please update submodules and try again.")
endif()


include_directories(${INCLUDE_DIRECTORIES} ${NumPy_INCLUDE_DIR} libklb/src)

add_subdirectory(pyklb)

add_subdirectory(libklb)
set(BINDINGS_JAVA       OFF)
set(build_tests         OFF)
set(SKIP_INSTALL_ALL    TRUE)
set(SKIP_INSTALL_EXPORT TRUE)
set(SKIP_TESTS          TRUE)
set_target_properties(klb bzip2 zlib PROPERTIES EXCLUDE_FROM_ALL TRUE)
