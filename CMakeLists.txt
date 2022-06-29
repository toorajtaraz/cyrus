cmake_minimum_required(VERSION 3.10)


PROJECT (cyrus LANGUAGES CXX CUDA)
set(OpenCV_DIR "${CMAKE_CURRENT_SOURCE_DIR}/submodules/build/")
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

#find cuda package and add include directories
find_package( CUDA REQUIRED )
include_directories( ${CUDA_INCLUDE_DIRS} )

# add_library(cyrus_lib STATIC
#   headers/helpers.h
# )

# set_target_properties(cyrus_lib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

set( NAME_SRC
  main.cu
  utils/helpers.cu
)

set( NAME_HEADERS
  headers/helpers.h
)

INCLUDE_DIRECTORIES( ${CMAKE_CURRENT_SOURCE_DIR}/headers )

link_directories( ${CMAKE_BINARY_DIR}/bin)

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)

#enable dynamic parallelism for cuda
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-rdc=true" )

add_executable( cyrus ${NAME_SRC} ${NAME_HEADERS} )
set_property(TARGET cyrus PROPERTY CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries( cyrus PRIVATE   -lcudadevrt -lcublas_device  ${OpenCV_LIBS} ${CUDA_LIBRARIES} ${CUDA_NVCC_FLAGS})