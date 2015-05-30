
cmake_minimum_required(VERSION 2.8.12)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}")

project(SkillInference)
include_directories(.)

find_package( Boost COMPONENTS program_options REQUIRED)
find_package( GSL REQUIRED)

include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${GSL_INCLUDE_DIRS})
include_directories(${Boost_INCLUDE_DIRS})

file(GLOB lib_srcs "src/*.cpp")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_executable(infer_skills ${lib_srcs}) 
target_link_libraries(infer_skills ${Boost_LIBRARIES} ${GSL_LIBRARIES})
