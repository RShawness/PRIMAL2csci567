cmake_minimum_required(VERSION 3.14)
set(CMAKE_CXX_STANDARD 14)
project(pybind)
add_subdirectory(pybind11)

include_directories("inc")
include_directories("src")


file(GLOB SOURCES "src/*.cpp")
list(FILTER SOURCES EXCLUDE REGEX "./src/pybinding_CBS.cpp")
list(FILTER SOURCES EXCLUDE REGEX "./src/pybind.cpp")
# file(GLOB SOURCES "inc/*.h")
# set(SOURCE "./src/python_test.py")
# add_executable(test_pybind_exe ${SOURCE})

set(Boost_INCLUDE_DIR /opt/homebrew/Cellar/boost/1.83.0/include)
set(Boost_LIBRARY_DIR /opt/homebrew/Cellar/boost/1.83.0/lib)
find_package(Boost REQUIRED COMPONENTS system filesystem)
include_directories(${Boost_INCLUDE_DIRS})

add_library(add STATIC ${SOURCES})
set_target_properties(add PROPERTIES POSITION_INDEPENDENT_CODE ON)

pybind11_add_module(cbs_py src/pybinding_CBS.cpp)
target_link_libraries(cbs_py PRIVATE add ${Boost_LIBRARIES})

