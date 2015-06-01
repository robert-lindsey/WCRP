
cmake_minimum_required(VERSION 2.8.12)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}")

project(WCRP)
include_directories(.)

find_package( Boost COMPONENTS program_options REQUIRED)
find_package( GSL REQUIRED)

include_directories(${CMAKE_SOURCE_DIR}/include ${GSL_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS})
file(GLOB lib_srcs "src/*.cpp")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_executable(cross_validation samples/cross_validation.cpp ${lib_srcs}) 
add_executable(find_skills samples/find_skills.cpp ${lib_srcs})

target_link_libraries(cross_validation ${Boost_LIBRARIES} ${GSL_LIBRARIES})
target_link_libraries(find_skills ${Boost_LIBRARIES} ${GSL_LIBRARIES})

