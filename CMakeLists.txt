cmake_minimum_required(VERSION 3.10)
PROJECT (cyrus LANGUAGES CXX CUDA)
set(OpenCV_DIR "${CMAKE_CURRENT_SOURCE_DIR}/submodules/build/")
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

#find cuda package and add include directories
find_package( CUDA REQUIRED )
include_directories( ${CUDA_INCLUDE_DIRS} )


set( NAME_SRC
  main.cpp
  utils/helpers.cpp
)

set( NAME_HEADERS
  headers/helpers.hpp
)

INCLUDE_DIRECTORIES( ${CMAKE_CURRENT_SOURCE_DIR}/headers )

link_directories( ${CMAKE_BINARY_DIR}/bin)

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)

add_executable( cyrus ${NAME_SRC} ${NAME_HEADERS} )
set_property(TARGET cyrus 
             PROPERTY CUDA_SEPARABLE_COMPILATION ON)


target_link_libraries( cyrus PRIVATE  ${OpenCV_LIBS})